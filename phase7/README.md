# Phase 7: prior case retrieval (PCR)

For every test query case, rank the candidate cases of the test pool and predict the 7 most likely to be cited.
Output: `prediction.json` = `{query id: [7 unique candidate ids, best first]}` inside `submission.zip`. Metric:
micro-F1@7 (primary), micro-precision@7 and micro-recall@7.

Rules followed:
- every neural experiment starts from our Phase 1 models (no other pretrained model, no LLM API);
- the fine-tuning techniques, BM25, the rankers and the losses are our own code;
- each split's pool is used only with that split's queries;
- test labels do not exist for us, and test text is never trained on (also not in mid-training).

## Method

| Stage | What | File |
|---|---|---|
| Lexical retrieval | Our own Okapi BM25 over word 1- to 5-grams (n >= 2 hashed), one sparse matrix per pool; each query's scores divided by its maximum. IL-PCR (Joshi et al. 2023) found word n-gram BM25 much stronger than pretrained transformers on Indian prior case retrieval. Gives candidates (BM25 top 100), hard negatives for training, and ranker features. | `pcr_data.py` |
| Other signals | Latest year mentioned in a case (about its decision year; a case decided after the query cannot be cited), and how often a candidate is cited by labelled queries of other splits (leave-one-out on train). | `pcr_data.py` |
| Lexical-only systems | (a) BM25 averaged over the n-gram orders; (b) a ranker on the BM25 scores + year + citation counts fitted on the training queries (dev stays an honest test), refitted on train + dev for the test predictions. | `pcr_data.lexical_submissions` |
| Mid-training | Next-token prediction on the case texts of the training and dev splits (queries + pools), not test. Not for the encoder-decoder. | `pcr_mid.py` |
| Case encoder (bi-encoder) | Our Phase 1 model reads a case in chunks of 512 tokens; each chunk -> token vectors -> mean pooling -> linear layer to 256 values -> length 1. A case = its chunk vectors (up to 24, spread over the whole judgment). | `pcr_model.py` |
| Training | Contrastive (DPR / ANCE style): 8 training queries per step, each with one cited case and 2 BM25 hard negatives; the other queries' cases are in-batch negatives; a case vector = average of 3 random chunks (different every step). InfoNCE over cosine / 0.05; cited cases never used as negatives. Variants: listwise (all cited cases in the batch at once), pairwise hinge, random negatives, last-token pooling. Gradient checkpointing per group of chunks. The weights of the epoch with the best dev micro-F1@7 (our model alone) are kept. | `pcr_train.py` |
| Fine-tuning techniques | Full fine-tuning, LoRMA (compulsory), QLoRA (4-bit NF4 frozen weights + LoRA), (IA)^3 (new: learned scaling vectors on activations), partial (top third of the blocks). Encoder-decoder: low-rank updates on its GRU weights with cuDNN kept (weights copied at every call); (IA)^3 scales the GRU outputs. | Phase 4 `cj_tech.py` |
| Case similarity | From the chunk vectors: mean (cosine of averaged vectors), maxsim (best-matching candidate chunk per query chunk, averaged), maxsim_sym, max (best chunk pair), first (first chunks only). All are computed for dev and test at the end of every run. | `pcr_model.aggregate` |
| Fusion (learning to rank) | Per query, BM25 top 100 plus our model's top 50 candidates; features (each standardised within the query): BM25 1-5-gram scores, their average and its rank, year features, citation counts, our model's mean / maxsim scores and ranks. Logistic regression (and coordinate ascent on the number of hits in the top 7, started from it). Measured by 5-fold cross-fitting on dev; the test ranker is fitted on all of dev (our model's scores on the training queries are too good to fit on). | `pcr_data.fused_submission` |
| Knowledge distillation | Teacher: the better of the full / LoRMA runs on dev (its case vectors of the training split are saved). Student: the same (mid-trained) model with half of its blocks, evenly spaced (encoder-decoder: half of its GRU layers), trained with 0.5 x InfoNCE + 0.5 x KL(teacher || student) between the teacher's and the student's similarity distributions over the batch's candidates (listwise; Margin-MSE also available). The same student without the teacher is trained for comparison. | `pcr_grid.run_kd` |
| Ensembles | Several runs: the average of their per-query standardised scores as features of the same ranker. | `pcr_grid.ensemble` |
| Final ranking (07/11, CPU) | Exact judgment dates from the header (DATE OF JUDGMENT, else CITATION year, else the latest year outside CITATOR INFO): candidates certainly decided after the query removed. Votes of the 20 most similar labelled queries (BM25 3-5-grams between query texts) for the cases they cite. Citation contexts: the 40 words before and after each `<CITATION>` mask in the query (masks are in every query) are short queries about one cited case; BM25 2-/3-gram scores of the best-matching context and of all contexts together (He et al. 2010, context-aware citation recommendation). Candidates from the 3/4/5-gram and averaged BM25 lists, the votes, the contexts and each model score. Our scores: per model the average of its 4 best runs (mean, maxsim, maxsim_sym), and the average over the four models. Citation strings in the headnotes (variants with / without): the query's own AIR citation listed in a candidate's CITATOR INFO, a candidate's citation written in the query, later cases that cited both; this uses the publisher's citator metadata and is reported separately. Rankers: logistic regression, pairwise logistic regression, coordinate ascent; 48 variants compared by 5-fold cross-fitting on dev; the best written as named submissions. | `pcr_final.py` |
| Time control | `finish_by`: each run measures its encoding and training speed, fewer chunks per case for the final encoding if needed, and trains only the steps that fit. | `pcr_train.py` |

## Files

- `pcr_data.py` (no torch): data, BM25, metric, prediction file, year / citation signals, rankers, fusion, explore.
- `pcr_model.py`: chunks, case encoding, case similarities, ranking and KD losses, students.
- `pcr_train.py`: one model + one technique (or a KD student); `comparison()` table.
- `pcr_grid.py`: `run_grid`, `run_variants`, `run_kd`, `best_run`, `ensemble`, `report` (one worker process per GPU).
- `pcr_mid.py`: mid-training.
- `pcr_final.py` (no torch): the final ranking over all runs' saved scores; `test_pcr_final.py`.
- Tests: `test_pcr_data.py` (no torch), `test_pcr_train.py` (tiny stand-in models; the notebook runs both first).

## Research basis

- Joshi et al. 2023, U-CREAT (ACL 2023): IL-PCR corpus (7,070 Indian Supreme Court cases, 1,182 queries, about
  6.8 citations per query); BM25 unigram 13.9, bigram 28.6, events + BM25 39.2 micro-F1@K; full-document BERT-style
  models about 9-17. https://arxiv.org/abs/2307.05260 (https://aclanthology.org/2023.acl-long.777)
- Robertson & Zaragoza 2009, BM25.
- Karpukhin et al. 2020, DPR: in-batch negatives + BM25 hard negatives. https://arxiv.org/abs/2004.04906
- Xiong et al. 2021, ANCE: hard negatives for dense retrieval. https://arxiv.org/abs/2007.00808
- Khattab & Zaharia 2020, ColBERT late interaction (maxsim; here at chunk level). https://arxiv.org/abs/2004.12832
- Hofstaetter et al. 2020, Margin-MSE cross-architecture distillation. https://arxiv.org/abs/2010.02666
- Liu et al. 2022, (IA)^3 / T-Few. https://arxiv.org/abs/2205.05638
- Dettmers et al. 2023, QLoRA; Bihany, Patel & Modi 2025, LoRMA.
- Metzler & Croft 2007, coordinate ascent for learning to rank.
- He et al. 2010 (WWW), context-aware citation recommendation: the text around a citation placeholder as the query.
- Gao et al. 2021, gradient caching / memory-efficient contrastive training (here: gradient checkpointing).
