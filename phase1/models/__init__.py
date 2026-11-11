"""Model registry: every architecture is built from (arch name, dict of hyperparameters)."""


def build_model(arch: str, model_cfg: dict, vocab_size: int):
    if arch == "decoder":
        from models.decoder import LMConfig, DecoderLM

        return DecoderLM(LMConfig(vocab_size=vocab_size, **model_cfg))
    if arch == "llama":
        from models.layers import ModernConfig
        from models.llama import LlamaLM

        return LlamaLM(ModernConfig(vocab_size=vocab_size, **model_cfg))
    if arch == "moe":
        from models.layers import ModernConfig
        from models.moe import MoELM

        return MoELM(ModernConfig(vocab_size=vocab_size, **model_cfg))
    if arch == "encdec":
        from models.encdec import EncDecConfig, EncDecLM

        return EncDecLM(EncDecConfig(vocab_size=vocab_size, **model_cfg))
    raise ValueError(f"unknown arch {arch!r}")


def init_weights(model, std=0.02):
    """GPT-2 style init."""
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
