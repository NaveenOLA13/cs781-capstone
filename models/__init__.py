"""Model registry: every architecture is built from (arch name, dict of hyperparameters).

To add architecture #2/#3/#4: put it in models/<name>.py exposing a config dataclass and a
model class whose forward(idx: LongTensor[B,T]) -> logits[B,T,V], then add one line below.
"""


def build_model(arch: str, model_cfg: dict, vocab_size: int):
    if arch == "decoder":
        from models.decoder import LMConfig, DecoderLM
        return DecoderLM(LMConfig(vocab_size=vocab_size, **model_cfg))
    if arch == "encdec":
        from models.encdec import EncDecConfig, EncDecLM
        return EncDecLM(EncDecConfig(vocab_size=vocab_size, **model_cfg))
    # elif arch == "hybrid":   from models.hybrid import ...
    raise ValueError(f"unknown arch {arch!r}")


def init_weights(model, std=0.02):
    """GPT-2 style init. PyTorch's default nn.Embedding init is N(0, 1); with the tied
    output head that gives huge initial logits (initial loss ~25 instead of ln(V)~9.4)."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)


def count_params(model) -> int:
    # tied embeddings are one Parameter object, so they are counted once
    return sum(p.numel() for p in model.parameters())