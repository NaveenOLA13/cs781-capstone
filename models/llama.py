"""Architecture #4: Llama-style dense decoder (Touvron et al. 2023, LLaMA).

RMSNorm + RoPE + grouped-query attention + dense SwiGLU feed-forward in every block.
"""
from models.layers import ModernConfig, DecoderBackbone, SwiGLU


class LlamaLM(DecoderBackbone):
    def __init__(self, cfg: ModernConfig):
        super().__init__(cfg, make_ffn=lambda: SwiGLU(cfg.d_model, cfg.d_ff))