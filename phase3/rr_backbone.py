"""Connect YOUR Phase 1 model and tokenizer to the Phase 3 code.

    from rr_backbone import make_backbone
    bb = make_backbone(model, tokenizer)        # finds everything itself and prints what it found

It works out:
  * encode(text) -> token ids, and the pad / bos / eos ids, from your tokenizer
  * how to get one hidden vector per token from your model (LM head skipped), trying in order:
      model.hidden_states(ids), model.encode(ids), the output of model.encoder while running
      model(ids, ids) or model(ids), model.encoder(ids), the input of the LM head in model(ids)
  * whether the model reads left-to-right (GPT, Llama, MoE, one-way RNN) -> pooling "last",
    or both ways (BiGRU / transformer encoder) -> pooling "mean"
  * that max_len tokens fit through the model

If nothing works it prints the model's structure; send a screenshot of it. You can also give
the way yourself, e.g. make_backbone(model, tokenizer, call=lambda m, ids, mask: m.encoder(m.embed(ids)))
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn

from rr_head import HiddenStates, SentenceClassifier, count_params, find_lm_head

SAMPLE = ("The appellant filed an appeal against the order of the High Court dated 2.3.2006 "
          "in ITA Number 50 of 2005 and the Tribunal allowed it.")
SHORT = "Leave granted."
PAD_NAMES = ("<pad>", "[PAD]", "<|pad|>", "<PAD>", "[pad]", "<|padding|>")
BOS_NAMES = ("<bos>", "<s>", "[BOS]", "<BOS>", "<|bos|>", "<|startoftext|>", "<|begin_of_text|>")
EOS_NAMES = ("<eos>", "</s>", "<|endoftext|>", "[EOS]", "<EOS>", "<|eos|>", "<|end_of_text|>")
ENCODER_NAMES = ("encoder", "enc", "encoder_stack", "encoder_layers", "src_encoder")
ENCODE_METHODS = ("encode", "encoder_forward", "forward_encoder", "encode_src", "get_encoder_output")


@dataclass
class Backbone:
    model: nn.Module
    encode: object
    pad_id: int = 0
    bos_id: int | None = None          # added to every sentence only if not None
    eos_id: int | None = None
    kind: str = "decoder"              # "decoder" = reads left-to-right, "encoder" = both ways
    pooling: str = "last"
    module_name: str | None = None
    mode: str = "pre_head"
    call: object = None
    route: str = ""
    d_model: int = 0
    vocab: int | None = None
    pad_effect: float = 0.0          # how much padding changes a sentence vector (0 = not at all)

    def classifier(self, n_labels: int, model: nn.Module | None = None, pooling: str | None = None,
                   **kw) -> SentenceClassifier:
        """SentenceClassifier on this backbone (or on `model`, e.g. a fresh copy of it)."""
        return SentenceClassifier(self.model if model is None else model, n_labels,
                                  pooling=pooling or self.pooling, d_model=self.d_model or None,
                                  module_name=self.module_name, mode=self.mode, call=self.call, **kw)


# ------------------------------------------------------------------------- tokenizer
def make_encode(tokenizer):
    """encode(text) -> list[int] from a tokenizer object (or a function text -> ids)."""
    if hasattr(tokenizer, "encode_as_ids"):          # sentencepiece
        f = tokenizer.encode_as_ids
    elif hasattr(tokenizer, "encode"):
        f = tokenizer.encode
    elif callable(tokenizer):
        f = tokenizer
    else:
        raise TypeError("tokenizer needs .encode(text), or pass encode=<function text -> list of ids>")

    def encode(text: str) -> list[int]:
        out = f(text)
        if isinstance(out, dict):
            out = out.get("input_ids", out.get("ids"))
        if hasattr(out, "ids") and not torch.is_tensor(out):   # tokenizers.Encoding
            out = out.ids
        if hasattr(out, "tolist"):
            out = out.tolist()
        out = list(out)
        if out and isinstance(out[0], (list, tuple)):
            out = list(out[0])
        return [int(x) for x in out]

    ids = encode(SAMPLE)
    if not ids:
        raise ValueError("the tokenizer returned no tokens for a normal sentence")
    return encode


def _lookup_id(tokenizer, kind: str, names) -> int | None:
    if tokenizer is None:
        return None
    for a in (f"{kind}_id", f"{kind}_token_id", f"{kind}_idx", f"{kind.upper()}_ID", f"{kind}_index"):
        v = getattr(tokenizer, a, None)
        if callable(v):
            try:
                v = v()
            except Exception:  # noqa: BLE001
                continue
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            return v
    dicts = []
    for a in ("special_tokens", "vocab", "stoi", "token2id", "token_to_idx", "str_to_id", "encoder"):
        d = getattr(tokenizer, a, None)
        if isinstance(d, dict):
            dicts.append(d)
    gv = getattr(tokenizer, "get_vocab", None)
    if callable(gv):
        try:
            dicts.append(gv())
        except Exception:  # noqa: BLE001
            pass
    fns = [getattr(tokenizer, m) for m in ("token_to_id", "piece_to_id", "convert_tokens_to_ids")
           if callable(getattr(tokenizer, m, None))]
    for name in names:
        for d in dicts:
            v = d.get(name)
            if isinstance(v, int) and v >= 0:
                return v
        for fn in fns:
            try:
                v = fn(name)
                unk = fn("<<surely-not-a-token-xyz>>")    # sentencepiece maps unknown pieces to unk
            except Exception:  # noqa: BLE001
                continue
            if isinstance(v, int) and v >= 0 and v != unk:
                return v
    return None


def special_ids(tokenizer) -> dict:
    return {"pad": _lookup_id(tokenizer, "pad", PAD_NAMES), "bos": _lookup_id(tokenizer, "bos", BOS_NAMES),
            "eos": _lookup_id(tokenizer, "eos", EOS_NAMES)}


# --------------------------------------------------------------------- hidden states
def _sub_call(name):
    return lambda m, ids, mask: m.get_submodule(name)(ids)


def _method_call(meth):
    return lambda m, ids, mask: getattr(m, meth)(ids)


def _two_arg_call(m, ids, mask):
    return m(ids, ids)


def _start_pos_call(m, ids, mask):
    return m(ids, 0)


def candidate_routes(model: nn.Module) -> list[tuple[str, dict]]:
    routes = []
    if callable(getattr(model, "hidden_states", None)):
        routes.append(("model.hidden_states(ids)", {}))
    base = set(dir(nn.Module))
    for meth in ENCODE_METHODS:
        if meth not in base and callable(getattr(model, meth, None)):
            routes.append((f"model.{meth}(ids)", dict(mode="output", call=_method_call(meth))))
    enc = sorted((n for n, _ in model.named_modules() if n and n.split(".")[-1].lower() in ENCODER_NAMES),
                 key=lambda n: n.count("."))
    for name in enc:
        routes.append((f"output of model.{name} while running model(ids, ids)",
                       dict(module_name=name, mode="module_out", call=_two_arg_call)))
        routes.append((f"output of model.{name} while running model(ids)",
                       dict(module_name=name, mode="module_out")))
        routes.append((f"model.{name}(ids)", dict(mode="output", call=_sub_call(name))))
    lm = find_lm_head(model)
    if lm:
        # module_name is given explicitly so a (failing) hidden_states method is not used instead
        routes.append((f"input of the LM head ({lm}) while running model(ids)",
                       dict(mode="pre_head", module_name=lm)))
        # nanoGPT computes logits for every position only when targets are given
        routes.append((f"input of the LM head ({lm}) while running model(ids, targets=ids)",
                       dict(mode="pre_head", module_name=lm, call=_two_arg_call)))
        # Meta-Llama style forward(tokens, start_pos)
        routes.append((f"input of the LM head ({lm}) while running model(ids, 0)",
                       dict(mode="pre_head", module_name=lm, call=_start_pos_call)))
    return routes


def _probe(model, kw, ids):
    """Hidden states for ids [B, T] through one route, or raise."""
    hs = HiddenStates(model, **kw)
    with torch.no_grad():
        h = hs(ids, torch.ones_like(ids))
    if not h.is_floating_point() or not torch.isfinite(h.float()).all():
        raise RuntimeError("hidden states are not finite floats")
    return h


def _first_working(model, routes, tokens, short, device):
    """First route that gives [1, T, d] hidden states that depend on the input."""
    errors = []
    for desc, kw in routes:
        try:
            h = _probe(model, kw, torch.tensor([tokens], device=device))
            h2 = _probe(model, kw, torch.tensor([short], device=device))
            if torch.allclose(h.float().mean(1), h2.float().mean(1), atol=1e-6):
                raise RuntimeError("output does not depend on the input text")
            return (desc, kw, h.size(-1)), errors
        except Exception as e:  # noqa: BLE001
            errors.append(f"  {desc}: {type(e).__name__}: {str(e)[:160]}")
    return None, errors


def _is_causal(model, kw, tokens, dev) -> bool:
    ids = torch.tensor([tokens], device=dev)
    k = max(1, len(tokens) // 2)
    full = _probe(model, kw, ids).float()
    pre = _probe(model, kw, ids[:, :k]).float()
    diff = (pre - full[:, :k]).abs().max().item()
    return diff <= 1e-3 * (1.0 + full.abs().max().item())


def describe(model: nn.Module, tokenizer=None) -> None:
    """Print what the Phase 3 code needs to know about a model (send a screenshot if asked)."""
    print(f"model class: {type(model).__module__}.{type(model).__name__}, "
          f"{count_params(model, False) / 1e6:.1f}M parameters")
    try:
        print(f"forward{inspect.signature(model.forward)}")
    except (TypeError, ValueError):
        pass
    print("top-level parts:")
    for n, c in model.named_children():
        try:
            sig = str(inspect.signature(c.forward))
        except (TypeError, ValueError):
            sig = "(?)"
        print(f"  {n:22s} {type(c).__name__:26s} {count_params(c, False) / 1e6:8.2f}M   forward{sig}")
    own = sorted(m for m in set(dir(type(model))) - set(dir(nn.Module))
                 if not m.startswith("_") and callable(getattr(model, m, None)))
    print("model methods:", own)
    lm = find_lm_head(model)
    print("LM head:", lm)
    if tokenizer is not None:
        meths = sorted(m for m in dir(tokenizer) if not m.startswith("_"))
        print(f"tokenizer: {type(tokenizer).__module__}.{type(tokenizer).__name__}; attributes: {meths[:60]}")


def vocab_size(model: nn.Module) -> int | None:
    """Token vocabulary size: the LM head's output size if there is one, else the largest
    nn.Embedding (an upper bound for valid token ids)."""
    lm = find_lm_head(model)
    if lm:
        return model.get_submodule(lm).out_features
    sizes = [m.num_embeddings for m in model.modules() if isinstance(m, nn.Embedding)]
    return max(sizes) if sizes else None


def make_backbone(model: nn.Module, tokenizer=None, encode=None, pad_id: int | None = None,
                  bos_id: int | None = None, eos_id: int | None = None, add_bos: bool = False,
                  add_eos: bool = False, kind: str | None = None, module_name: str | None = None,
                  mode: str | None = None, call=None, max_len: int = 128, verbose: bool = True) -> Backbone:
    """All checks run on the CPU in float32 (a bad token id on the GPU would break the session);
    the model is put back on its device and dtype afterwards."""
    say = print if verbose else (lambda *a, **k: None)
    encode = encode or make_encode(tokenizer)
    sp = special_ids(tokenizer)
    pad_id = pad_id if pad_id is not None else (sp["pad"] if sp["pad"] is not None else 0)
    bos_id = bos_id if bos_id is not None else sp["bos"]
    eos_id = eos_id if eos_id is not None else sp["eos"]
    if add_bos and bos_id is None:
        raise ValueError("add_bos=True but no bos id was found; pass bos_id=...")
    if add_eos and eos_id is None:
        raise ValueError("add_eos=True but no eos id was found; pass eos_id=...")

    tokens = encode(SAMPLE)
    short = encode(SHORT) or tokens[:2]
    if len(tokens) < 4:
        tokens = (tokens * 4)[:8]
    vocab = vocab_size(model)
    used = {"pad": pad_id, "bos": bos_id if add_bos else None, "eos": eos_id if add_eos else None,
            "largest sample token": max(tokens + short)}
    if vocab is not None:
        bad = {k: v for k, v in used.items() if v is not None and v >= vocab}
        if bad:
            raise ValueError(f"token ids {bad} are >= the model's vocabulary size {vocab}: "
                             "is this the tokenizer the model was trained with?")

    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    was_training = model.training
    model.eval()
    model.cpu().float()
    cpu = torch.device("cpu")
    try:
        if call is not None or module_name is not None or mode is not None:
            m = mode or ("output" if call is not None and module_name is None
                         else "module_out" if module_name else "pre_head")
            routes = [("the way you gave", dict(module_name=module_name, mode=m, call=call))]
        else:
            routes = candidate_routes(model)
        chosen, errors = _first_working(model, routes, tokens, short, cpu)
        pdev = cpu
        if chosen is None and dev.type == "cuda":
            # some models keep tensors on the GPU outside their parameters; try there too
            model.to(dev)
            pdev = dev
            chosen, errors_gpu = _first_working(model, routes, tokens, short, dev)
            errors += ["  (again on the GPU)"] + errors_gpu
        if chosen is None:
            print("Could not get hidden states from the model. Tried:")
            print("\n".join(errors) if errors else "  (nothing to try: no encoder, LM head or hidden_states method found)")
            print()
            describe(model, tokenizer)
            raise RuntimeError("no way found to get hidden states; send a screenshot of the output above")
        desc, kw, d_model = chosen

        causal = _is_causal(model, kw, tokens, pdev)
        if causal and kw.get("mode") != "pre_head" and "hidden_states" not in desc and len(routes) > 1:
            # a decoder: prefer the states right before the LM head (after the final norm)
            for d2, kw2 in routes:
                if kw2.get("mode") == "pre_head":
                    try:
                        _probe(model, kw2, torch.tensor([tokens], device=pdev))
                        desc, kw = d2, kw2
                        break
                    except Exception:  # noqa: BLE001
                        continue
        kind = kind or ("decoder" if causal else "encoder")
        pooling = "last" if kind == "decoder" else "mean"

        long_ids = torch.tensor([(tokens * (max_len // len(tokens) + 1))[:max_len]] * 2, device=pdev)
        try:
            _probe(model, kw, long_ids)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"the model cannot take {max_len} tokens ({type(e).__name__}: {e}); "
                               "use a smaller max_len") from e

        # how much padding changes a short sentence's vector when batched with a long one
        a = torch.tensor([short], device=pdev)
        b = torch.full((2, len(tokens)), pad_id, dtype=torch.long, device=pdev)
        b[0, :len(short)] = a[0]
        b[1] = torch.tensor(tokens, device=pdev)
        ha = _probe(model, kw, a).float()[0]
        hb = _probe(model, kw, b).float()[0, :len(short)]
        if pooling == "last":
            pad_diff = (ha[-1] - hb[-1]).abs().max().item()
        else:
            pad_diff = (ha.mean(0) - hb.mean(0)).abs().max().item()
        scale = ha.abs().max().item() + 1e-9
    finally:
        model.to(dev)
        if dtype != torch.float32:
            model.to(dtype)
        model.train(was_training)

    bb = Backbone(model=model, encode=encode, pad_id=pad_id, bos_id=bos_id if add_bos else None,
                  eos_id=eos_id if add_eos else None, kind=kind, pooling=pooling,
                  module_name=kw.get("module_name"), mode=kw.get("mode", "pre_head"), call=kw.get("call"),
                  route=desc, d_model=d_model, vocab=vocab,
                  pad_effect=pad_diff / scale)
    say(f"[backbone] hidden states : {desc}")
    say(f"[backbone] model type    : {kind} ({'reads left-to-right' if causal else 'reads both ways'})"
        f" -> pooling '{pooling}', hidden size d = {d_model}")
    say(f"[backbone] tokenizer     : pad_id={pad_id}, bos={bos_id}, eos={eos_id} "
        f"(added to sentences: bos={'yes' if add_bos else 'no'}, eos={'yes' if add_eos else 'no'}); "
        f"sample sentence -> {len(encode(SAMPLE))} tokens; vocabulary {vocab}")
    say(f"[backbone] parameters    : {count_params(model, False) / 1e6:.1f}M; {max_len} tokens fit through the model")
    note = ("none" if pad_diff <= 1e-4 * scale else
            f"{pad_diff / scale:.1%} of the vector size -> training will batch only sentences of "
            "exactly the same length, so no sentence is ever padded")
    say(f"[backbone] padding effect: {note}")
    return bb