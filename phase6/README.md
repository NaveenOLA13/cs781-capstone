# Phase 6: abstractive summarization of legal documents

For every test document, write a summary with our own Phase 1 models. The output is `[{"id", "summary"}]` inside
`submission.zip` (`prediction.json`). The metric is BERTScore F1 (primary), with ROUGE-1 and ROUGE-L F1 as secondary
metrics.

Rules followed:
- every run starts from our Phase 1 models;
- only the competition data is used (training documents and summaries; test documents only for writing their
  summaries);
- the held-out part of the training data is never trained on;
- all techniques are our own code.

## Method (main grid: 4 models x LoRA, LoRMA, QLoRA, adapter + knowledge distillation)

| Stage | What | File |
|---|---|---|
| Mid-training | Next-token prediction on the training documents and summaries (domain and summary style), with a time limit. Not for the encoder-decoder. | `summ_mid.py` |
| Long documents (divide and conquer) | Each document is cut into windows that fit the model: 763 tokens for the decoder-only models with a 1,024-token limit, 512 for the encoder-decoder. Each reference-summary sentence goes to the window holding the document sentence most similar to it (content-word overlap). A window's target is its sentences in summary order. Windows with no sentence are kept 20% of the time with an empty target. | `summ_data.py` |
| Fine-tuning | Decoder-only models: `[eos] window " Summary:" target [eos]`, loss on the target only. Encoder-decoder: the encoder reads the window, the decoder writes the target. Cross-entropy with label smoothing 0.1. Techniques: LoRA, LoRMA, QLoRA (4-bit NF4 frozen weights), adapter. On the encoder-decoder the low-rank techniques touch only its linear layers. | `summ_train.py`, `summ_model.py`, Phase 4 `cj_tech.py` |
| Knowledge distillation | Teacher: the better of LoRA / LoRMA on held-out ROUGE. Student: the same mid-trained model with half its blocks, evenly spaced (shrink and fine-tune). Loss: 0.5 x gold + 0.5 x T^2 KL(teacher, student) on the next-word distributions, T = 2. The same student without the teacher is trained for comparison. | `summ_grid.run_kd` |
| Generation | Every window is summarised twice on the held-out documents, by beam search (2 beams, length-normalised scores) and by top-p sampling (p 0.9, temperature 0.7); repeated 3-word sequences are banned in both. The decoding with the better held-out ROUGE writes the test summaries. The summary length is predicted from the document length (log-linear fit on training data) and shared among the windows. The parts are joined; repeated sentences and an unfinished last sentence are removed; the text is cut near the predicted length. | `summ_model.Generator`, `summ_data.postprocess` |
| Choosing runs | ROUGE-1/2/L F1 on held-out training documents (our own scorer), plus length, repeated trigrams, and the share of word pairs not in the document. | `summ_data.score_all`, `summ_grid.report` |
| Time control | `finish_by`: every run measures its training and generation speed and trains only the steps that fit, so summaries and submissions are always written. The notebook stops at once without a GPU. | `summ_train.py` |

## Files

- `summ_data.py` (no torch): data, sentences, windows, alignment, length model, scores, post-processing, submission.
- `summ_model.py`: the Phase 1 model as a generator, batches, loss with KD, techniques, students and teachers, batched
  beam search.
- `summ_train.py`: one model + one technique (or a KD student).
- `summ_grid.py`: `run_grid` (one worker per GPU), `run_kd`, `report`, `best_run`.
- `summ_mid.py`: mid-training.
- `phase6_run.ipynb`: the notebook for one model.
- Tests: `test_summ_data.py` (no torch), `test_summ_train.py` (tiny stand-in models; the notebook runs both first).

## Research basis

- **Shukla et al. 2022**, legal case document summarization: chunk-based fine-tuning for Indian Supreme Court
  judgments (IN-Abs: 7,030 / 100 documents, about 4,400-token documents and 1,050-token summaries). Summary sentences
  are mapped to the most similar chunk. Best published IN-Abs scores (large pretrained models): ROUGE-1 about 0.49,
  BERTScore about 0.85. https://arxiv.org/abs/2210.07544
- **Gidiotis & Tsoumakas 2020**, divide and conquer for long documents: split the document, align summary parts to
  the pieces by ROUGE, train on the pieces, join the outputs. https://arxiv.org/abs/2004.06190
- **Pilault et al. 2020**: a decoder-only transformer LM conditioned on (selected) document text writes the summary.
  https://arxiv.org/abs/1909.03186
- **Shleifer & Rush 2020**, summarization distillation: shrink and fine-tune (copy teacher layers into a smaller
  student, then train); pseudo-labels and logit KD as alternatives. https://arxiv.org/abs/2010.13002
- **Kim & Rush 2016**: sequence-level and word-level knowledge distillation for sequence generation.
