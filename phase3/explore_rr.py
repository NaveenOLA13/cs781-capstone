"""Explore the Phase 3 (rhetorical role) data: sizes, label balance, sentence lengths, where each role occurs in a
judgment, and which role follows which."""

import collections
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

R = os.environ.get("RR_DIR", "/kaggle/working/data/Phase_3_RR")
OUT = os.environ.get("RR_OUT", "/kaggle/working/rr_viz")
os.makedirs(OUT, exist_ok=True)
if not os.path.isdir(R):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="Exploration-Lab/CS-781-Capstone",
        repo_type="dataset",
        local_dir=os.path.dirname(R),
        allow_patterns=["Phase_3_RR/*"],
    )


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def records(obj):
    if isinstance(obj, dict):
        return [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in obj.items()]
    return obj


# --- load
docs = []
for p in sorted(glob.glob(f"{R}/**/*.json", recursive=True)):
    name = os.path.basename(p)
    if "label_vocab" in name:
        continue
    domain = name.split("_")[0].upper()
    split = "test" if "test" in name.lower() else ("dev" if "dev" in name.lower() else "train")
    for r in records(load(p)):
        sents = r.get("text") or r.get("sentences") or []
        docs.append(dict(split=split, domain=domain, id=str(r.get("id")), sents=sents, labels=r.get("labels")))

vp = glob.glob(f"{R}/**/label_vocab.json", recursive=True)
vocab = load(vp[0]) if vp else None
if isinstance(vocab, dict):
    if all(isinstance(v, int) for v in vocab.values()):
        LABELS = sorted(vocab, key=vocab.get)
    elif all(str(k).isdigit() for k in vocab):
        LABELS = [vocab[k] for k in sorted(vocab, key=int)]
    else:
        LABELS = list(vocab)
elif isinstance(vocab, list):
    LABELS = list(vocab)
else:
    LABELS = sorted({l for d in docs if d["labels"] for l in d["labels"]})
print("label_vocab:", vocab, "\n")

lab = [d for d in docs if d["labels"]]
DOMAINS = sorted({d["domain"] for d in docs})

# --- overview
print(
    f"{'split':6s} {'domain':6s} {'docs':>5s} {'sentences':>10s} {'sent/doc (median, max)':>24s} {'len mismatch':>13s}"
)
for split in ("train", "dev", "test"):
    for dom in DOMAINS:
        ds = [d for d in docs if d["split"] == split and d["domain"] == dom]
        if not ds:
            continue
        n = [len(d["sents"]) for d in ds]
        bad = sum(1 for d in ds if d["labels"] is not None and len(d["labels"]) != len(d["sents"]))
        print(f"{split:6s} {dom:6s} {len(ds):5d} {sum(n):10d} {int(np.median(n)):>16d}, {max(n):<6d} {bad:13d}")
unknown = collections.Counter(l for d in lab for l in d["labels"] if l not in LABELS)
if unknown:
    print("labels not in label_vocab:", dict(unknown))

# --- label table
cnt = {dom: collections.Counter(l for d in lab if d["domain"] == dom for l in d["labels"]) for dom in DOMAINS}
tot = collections.Counter(l for d in lab for l in d["labels"])
order = sorted(LABELS, key=lambda l: -tot[l])
print(f"\n{'label':24s}" + "".join(f"{dom:>14s}" for dom in DOMAINS) + f"{'all':>14s}")
for l in order:
    row = "".join(f"{cnt[dom][l]:7d} ({100 * cnt[dom][l] / max(1, sum(cnt[dom].values())):4.1f}%)" for dom in DOMAINS)
    print(f"{l:24s}{row}{tot[l]:7d} ({100 * tot[l] / max(1, sum(tot.values())):4.1f}%)")


def show(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=130)
    plt.show()


# --- 1. label distribution
fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 1.5))
y = np.arange(len(order))
h = 0.8 / max(1, len(DOMAINS))
for i, dom in enumerate(DOMAINS):
    share = [100 * cnt[dom][l] / max(1, sum(cnt[dom].values())) for l in order]
    ax.barh(y + i * h, share, height=h, label=dom)
ax.set_yticks(y + h * (len(DOMAINS) - 1) / 2)
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("% of labelled sentences in the domain (train + dev)")
ax.set_title("Rhetorical role distribution by domain")
ax.legend()
show(fig, "1_label_distribution")

# --- 2. sentences per document
fig, ax = plt.subplots(figsize=(8, 3.5))
for dom in DOMAINS:
    ax.hist([len(d["sents"]) for d in docs if d["domain"] == dom], bins=30, alpha=0.6, label=dom)
ax.set_xlabel("sentences per document (all splits)")
ax.set_ylabel("documents")
ax.set_title("Document length")
ax.legend()
show(fig, "2_sentences_per_doc")

# --- 3. sentence length
words = np.array([len(s.split()) for d in docs for s in d["sents"]])
pct = {p: int(np.percentile(words, p)) for p in (50, 90, 95, 99)}
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(np.clip(words, 0, 200), bins=50)
for p, v in pct.items():
    ax.axvline(min(v, 200), linestyle="--", linewidth=1)
    ax.text(min(v, 200), ax.get_ylim()[1] * 0.9, f" p{p}={v}", fontsize=8)
ax.set_xlabel("words per sentence (clipped at 200)")
ax.set_ylabel("sentences")
ax.set_title("Sentence length")
show(fig, "3_sentence_length")
print("\nwords per sentence percentiles:", pct, " max:", int(words.max()))

# --- 4. where each role occurs in a judgment
B = 10
pos = np.zeros((len(order), B))
idx = {l: i for i, l in enumerate(order)}
for d in lab:
    n = len(d["labels"])
    for j, l in enumerate(d["labels"]):
        if l in idx:
            pos[idx[l], min(int(j / n * B), B - 1)] += 1
pos_norm = pos / np.maximum(pos.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.5))
im = ax.imshow(pos_norm, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xticks(range(B))
ax.set_xticklabels([f"{10 * i}-{10 * (i + 1)}%" for i in range(B)], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("relative position in the judgment")
ax.set_title("Where each role occurs (each row sums to 1)")
fig.colorbar(im, ax=ax)
show(fig, "4_role_position")

# --- 5. which role follows which
T = np.zeros((len(order), len(order)))
for d in lab:
    for a, b in zip(d["labels"], d["labels"][1:]):
        if a in idx and b in idx:
            T[idx[a], idx[b]] += 1
T_norm = T / np.maximum(T.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = ax.imshow(T_norm, cmap="magma")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8)
ax.set_xlabel("next sentence's role")
ax.set_ylabel("this sentence's role")
ax.set_title("Role transitions (each row sums to 1)")
fig.colorbar(im, ax=ax)
show(fig, "5_role_transitions")
same = np.trace(T) / max(T.sum(), 1)
print(f"\nnext sentence has the same role as the current one: {same:.1%}")
runs = [sum(1 for a, b in zip(d["labels"], d["labels"][1:]) if a != b) + 1 for d in lab if d["labels"]]
print(f"role blocks per document: median {int(np.median(runs))}")

# --- a sample document
d = next((d for d in lab if d["split"] == "train"), lab[0] if lab else None)
if d:
    print(f"\nsample document {d['id']} ({d['domain']}), first 15 sentences:")
    for s, l in list(zip(d["sents"], d["labels"]))[:15]:
        print(f"  [{l:20s}] {s[:110]}")

json.dump(
    {
        "labels": order,
        "domains": DOMAINS,
        "counts": {dom: dict(cnt[dom]) for dom in DOMAINS},
        "docs": {
            f"{sp}/{dom}": [len(x["sents"]) for x in docs if x["split"] == sp and x["domain"] == dom]
            for sp in ("train", "dev", "test")
            for dom in DOMAINS
        },
        "word_percentiles": pct,
        "position": pos.tolist(),
        "transitions": T.tolist(),
        "same_role_next": same,
    },
    open(f"{OUT}/rr_summary.json", "w"),
    indent=1,
)
print(f"\ncharts and rr_summary.json saved in {OUT}")
