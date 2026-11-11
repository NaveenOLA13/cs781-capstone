"""Generates 128-word continuations for every test prompt -> prediction.json + submission.zip."""

import argparse
import json
import os
import time
import zipfile

import torch

from bpe import BPETokenizer
from models import build_model

TEST_HF = "datasets/Exploration-Lab/CS-781-Capstone/Phase_1_Pretraining/Test/test.json"


def load_prompts(path):
    if path.startswith("datasets/"):
        from huggingface_hub import HfFileSystem

        with HfFileSystem().open(path) as f:
            return json.load(f)
    return json.load(open(path))


def sample_next(logits, seq, a):
    """logits: (B, V) for the next position; seq: (B, T) tokens so far."""
    logits = logits.float()
    if a.repetition_penalty != 1.0:
        seen = torch.gather(logits, 1, seq)
        seen = torch.where(seen > 0, seen / a.repetition_penalty, seen * a.repetition_penalty)
        logits = logits.scatter(1, seq, seen)
    if a.temperature <= 0:
        return logits.argmax(-1)
    logits = logits / a.temperature
    if a.top_k > 0:
        kth = torch.topk(logits, a.top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0 < a.top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        cp = sl.softmax(-1).cumsum(-1)
        remove = cp - sl.softmax(-1) > a.top_p
        sl = sl.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, si, sl)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(1)


@torch.no_grad()
def generate_batch(model, prompts_ids, eos_id, max_seq_len, a, tok, device):
    L = min(len(p) for p in prompts_ids)
    x = torch.tensor([p[-L:] for p in prompts_ids], dtype=torch.long, device=device)
    B = x.size(0)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    new = [[] for _ in range(B)]
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    seq2seq = hasattr(model, "encode")
    if seq2seq:
        with amp:
            enc_out, h = model.encode(x)
        cur = x[:, -1:]
    for step in range(a.max_new_tokens):
        ctx = x[:, -max_seq_len:]
        with amp:
            if seq2seq:
                logits, h = model.decode(cur, h, enc_out)
                logits = logits[:, -1, :]
            else:
                logits = model(ctx)[:, -1, :]
        if a.ban_eos:
            logits[:, eos_id] = float("-inf")
        nxt = sample_next(logits, ctx, a)
        nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
        done |= nxt == eos_id
        for i, t in enumerate(nxt.tolist()):
            if t != eos_id:
                new[i].append(t)
        x = torch.cat([x, nxt[:, None]], dim=1)
        cur = nxt[:, None]
        if step % 16 == 15:
            if all(done[i] or len(tok.decode(new[i]).split()) >= a.words for i in range(B)):
                break
    return [" ".join(tok.decode(n).split()[: a.words]) for n in new]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tok/bpe12k/tokenizer.json")
    ap.add_argument("--test", default=TEST_HF)
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=600, help="upper bound; stops at 128 words")
    ap.add_argument(
        "--prompt_tokens", type=int, default=0, help="keep last N prompt tokens (0 = auto: encdec src_len, decoder 768)"
    )
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--ban_eos", type=int, default=1, help="1 = never emit <eos> (always reach 128 words)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only first N prompts (quick test)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["arch"], cfg["model"], ck["vocab_size"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    max_seq_len = cfg["model"]["max_seq_len"]
    tok = BPETokenizer.load(a.tokenizer)
    eos = tok.token_to_id("<eos>")
    assert tok.get_vocab_size() == ck["vocab_size"], "tokenizer does not match the checkpoint!"

    prompts = load_prompts(a.test)
    ids = list(prompts.keys())[: a.limit or None]
    keep = a.prompt_tokens or cfg["model"].get("src_len") or (max_seq_len - 256)
    enc = {k: tok.encode(prompts[k]["text"])[-keep:] for k in ids}
    order = sorted(ids, key=lambda k: len(enc[k]))
    print(f"[gen] {len(ids)} prompts, keep last {keep} prompt tokens, device={device}")

    preds, t0 = {}, time.time()
    for b in range(0, len(order), a.batch_size):
        batch = order[b : b + a.batch_size]
        outs = generate_batch(model, [enc[k] for k in batch], eos, max_seq_len, a, tok, device)
        for k, o in zip(batch, outs):
            preds[k] = {"text": o}
        print(f"[gen] {len(preds)}/{len(ids)}  ({time.time() - t0:.0f}s)", flush=True)

    preds = {k: preds[k] for k in ids}
    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(a.out, "prediction.json")
    json.dump(preds, open(pj, "w"), ensure_ascii=False, indent=1)
    with zipfile.ZipFile(os.path.join(a.out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    json.dump(vars(a), open(os.path.join(a.out, "gen_args.json"), "w"), indent=1)
    n_words = [len(v["text"].split()) for v in preds.values()]
    print(
        f"[gen] wrote {a.out}/submission.zip | {len(preds)} ids | words min/avg "
        f"{min(n_words)}/{sum(n_words) / len(n_words):.1f}"
    )
    print("[gen] example:", preds[ids[0]]["text"][:300])


if __name__ == "__main__":
    main()
