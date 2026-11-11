# Phase 1: pretraining four language models from scratch

## Pipeline

| Step | What | File |
|---|---|---|
| Corpus | streams the four sources (acts, Supreme Court, High Courts, tribunals) and cleans the text | `corpus.py` |
| Tokenizer | byte-level BPE written from scratch, 12,000 tokens, trained on a balanced sample of the corpus | `bpe.py`, `train_tokenizer.py`, `tok/bpe12k/tokenizer.json` |
| Token files | each source tokenized once into `{source}_train.bin` / `{source}_val.bin` (uint16, documents separated by `<eos>`; whole documents go to validation) | `prepare_data.py` |
| Sampling | random 1,024-token windows; each row picks a source with the configured weight (acts 0.04, Supreme Court 0.07, High Courts 0.55, tribunals 0.34) | `data.py` |
| Training | next-token prediction (the encoder-decoder: first 512 tokens to the encoder, loss on the next 512), AdamW, warmup + cosine decay, fp16, gradient clipping, 65,536 tokens per step, best checkpoint by validation loss, stops before Kaggle's 12-hour limit and resumes from `last.pt` | `train.py` |
| Generation | continuations of the test prompts for the Phase 1 submission | `generate.py` |

## Architectures (`models/`, settings in `configs/`)

| Model | File | Size | Design |
|---|---|---|---|
| Decoder-only | `decoder.py` | 80M | 10 blocks, d=768, 12 heads, GELU feed-forward 3,072, sinusoidal positions, pre-norm |
| Llama-style | `llama.py`, `layers.py` | 85M | 12 blocks, d=768, 12 heads, 4 key/value heads (grouped-query attention), RoPE, RMSNorm, SwiGLU 2,048 |
| Mixture of experts | `moe.py`, `layers.py` | 214M (72M per token) | 10 blocks, d=768, 8 SwiGLU experts of 1,024, top-2 routing, load-balancing loss 0.01 |
| Encoder-decoder | `encdec.py` | 86M | embeddings 512, 2-layer bidirectional GRU encoder (1,024 per direction), 2-layer GRU decoder with dot-product attention |

## Running (from this folder)

```
python train_tokenizer.py --vocab_size 12000 --docs_per_source 2000 --stride 5 --out tok/bpe12k
python prepare_data.py --out data/v1 --plan "acts=all:1,supreme_court=all:1,high_courts=300M:5,tribunals=200M:5"
python train.py --config configs/llama_85m.yaml                       # set data.dir in the config
python train.py --config configs/llama_85m.yaml --resume runs/llama_85m_v1/last.pt
python generate.py --ckpt runs/llama_85m_v1/best.pt --out sub/llama
```
