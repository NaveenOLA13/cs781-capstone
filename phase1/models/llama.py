"""Architecture 4: Llama-style decoder (Touvron et al. 2023)."""

from models.layers import ModernConfig, DecoderBackbone, SwiGLU


class LlamaLM(DecoderBackbone):
    def __init__(self, cfg: ModernConfig):
        super().__init__(cfg, make_ffn=lambda: SwiGLU(cfg.d_model, cfg.d_ff))
