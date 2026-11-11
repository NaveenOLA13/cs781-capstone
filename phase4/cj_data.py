"""Phase 4 (court judgment prediction, CJPE): data, tokenising, metrics and the submission file."""

from __future__ import annotations

import glob
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import zipfile
from collections import Counter

DEFAULT_ROOT = "/kaggle/working/data/Phase_4_CJPE"
HF_REPO = "Exploration-Lab/CS-781-Capstone"
TEXT_KEYS = ("text", "document", "case_text", "content", "judgment", "facts", "body")
LABEL_KEYS = ("label", "labels", "y", "target", "outcome", "decision", "judgment_label", "class")
ID_KEYS = ("id", "case_id", "doc_id", "name", "uid")
WINDOWS = ("tail", "head", "head_tail")


# --- loading
def download(root: str = DEFAULT_ROOT) -> str:
    """Downloads Phase_4_CJPE from the course's Hugging Face dataset if the folder is missing."""
    if not os.path.isdir(root):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=os.path.dirname(root),
            allow_patterns=[os.path.basename(root) + "/*"],
        )
    return root


def _records(obj) -> list[dict]:
    if isinstance(obj, dict):
        for k in ("data", "records", "cases", "examples"):
            if isinstance(obj.get(k), list):
                return obj[k]
        return [dict(v, id=v.get("id", k)) if isinstance(v, dict) else {"id": k, "text": v} for k, v in obj.items()]
    if not isinstance(obj, list):
        raise ValueError(f"expected a list or dict of cases, got {type(obj).__name__}")
    return obj


def _pick(keys, names):
    return next((n for n in names if n in keys), None)


def to_label(v) -> int:
    """0 / 1 from 0, 1, True, "0", "1", 0.0, 1.0 or a one-item list of these."""
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return int(v)
    if isinstance(v, str) and v.strip() in ("0", "1", "0.0", "1.0"):
        return int(float(v.strip()))
    raise ValueError(f"label {v!r} is not 0 or 1")


def load_split(path: str, need_label: bool, read_label: bool = True) -> list[dict]:
    """Cases of one file. read_label=False ignores any label field (used for test.json: its labels, if the file has
    any, are never read, so they cannot influence training, selection or thresholds)."""
    with open(path, encoding="utf-8") as f:
        recs = _records(json.load(f))
    if not recs:
        raise ValueError(f"{path}: no cases")
    keys = set().union(*(r.keys() for r in recs[:100]))
    tk, lk, ik = _pick(keys, TEXT_KEYS), _pick(keys, LABEL_KEYS), _pick(keys, ID_KEYS)
    if tk is None or ik is None:
        raise ValueError(f"{path}: no text field ({TEXT_KEYS}) or id field ({ID_KEYS}) in {sorted(keys)}")
    if need_label and lk is None:
        raise ValueError(f"{path}: no label field ({LABEL_KEYS}) in {sorted(keys)}")
    out = []
    for r in recs:
        t = r.get(tk, "")
        t = " ".join(map(str, t)) if isinstance(t, list) else ("" if t is None else str(t))
        lab = to_label(r[lk]) if (read_label and lk is not None and r.get(lk) is not None) else None
        if need_label and lab is None:
            raise ValueError(f"{path}: case {r.get(ik)!r} has no label")
        out.append(dict(id=str(r[ik]), text=t, label=lab))
    return out


def find_files(root: str) -> dict[str, str]:
    files = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        name = os.path.basename(p).lower()
        split = (
            "test"
            if "test" in name
            else "dev"
            if ("dev" in name or "val" in name)
            else ("train" if "train" in name else None)
        )
        if split is not None:
            if split in files:
                raise ValueError(f"two {split} files: {files[split]} and {p}")
            files[split] = p
    missing = {"train", "dev", "test"} - set(files)
    if missing:
        raise ValueError(f"no {sorted(missing)} file found under {root}")
    return files


def _key(d: dict) -> str:
    return hashlib.md5(d["text"].encode("utf-8")).hexdigest()


def clean_splits(data: dict[str, list[dict]]) -> dict:
    """Removes repeated cases and overlaps between splits (in place) and returns what was removed: - test cases (same
    id or same text) from dev and train."""
    test_ids = {d["id"] for d in data["test"]}
    test_keys = {_key(d) for d in data["test"]}
    rep = {}
    seen_train: set[str] = set()
    for split in ("train", "dev"):
        kept, n_test, n_dup, n_train = [], 0, 0, 0
        seen: set[str] = set()
        for d in data[split]:
            k = _key(d)
            if k in test_keys or d["id"] in test_ids:
                n_test += 1
            elif split == "dev" and k in seen_train:
                n_train += 1
            elif k in seen:
                n_dup += 1
            else:
                seen.add(k)
                kept.append(d)
        rep[split] = dict(
            before=len(data[split]),
            after=len(kept),
            test_cases_removed=n_test,
            repeats_removed=n_dup,
            train_cases_removed=n_train,
        )
        data[split] = kept
        if split == "train":
            seen_train = seen
    return rep


def load_data(root: str = DEFAULT_ROOT, fetch: bool = True, clean: bool = True, log=None) -> dict[str, list[dict]]:
    """{"train", "dev", "test"}; test labels are never read. clean=True: see clean_splits."""
    if fetch:
        download(root)
    files = find_files(root)
    data = {
        s: load_split(files[s], need_label=(s != "test"), read_label=(s != "test")) for s in ("train", "dev", "test")
    }
    dup = [k for k, c in Counter(d["id"] for d in data["test"]).items() if c > 1]
    if dup:
        raise ValueError(f"test.json has repeated ids, e.g. {dup[:3]}")
    if clean:
        rep = clean_splits(data)
        if log is not None:
            for s, r in rep.items():
                log(
                    f"[data] {s}: {r['before']} -> {r['after']} cases (removed {r['test_cases_removed']} test cases, "
                    f"{r['repeats_removed']} repeats"
                    + (f", {r['train_cases_removed']} also in train" if s == "dev" else "")
                    + ")"
                )
    return data


def summary(data: dict[str, list[dict]]) -> str:
    lines = [f"{'split':6s} {'cases':>7s} {'label 0':>9s} {'label 1':>9s} {'words: median':>14s} {'max':>8s}"]
    for s, ds in data.items():
        c = Counter(d["label"] for d in ds)
        w = sorted(len(d["text"].split()) for d in ds)
        lines.append(f"{s:6s} {len(ds):7d} {c.get(0, 0):9d} {c.get(1, 0):9d} {w[len(w) // 2]:14d} {w[-1]:8d}")
    return "\n".join(lines)


# --- tokenising
def tail_text(text: str, n_chars: int) -> str:
    """The last n_chars characters, starting at a word boundary (no half word at the cut)."""
    if n_chars >= len(text):
        return text
    part = text[-n_chars:]
    cut = part.find(" ")
    return part[cut + 1 :] if 0 <= cut < len(part) - 1 else part


def head_text(text: str, n_chars: int) -> str:
    """The first n_chars characters, ending at a word boundary."""
    if n_chars >= len(text):
        return text
    part = text[:n_chars]
    cut = part.rfind(" ")
    return part[:cut] if cut > 0 else part


def _grow(text, encode, n_tokens, take_part, keep):
    take = int(n_tokens * 6) + 64
    while True:
        part = take_part(text, take)
        ids = [int(x) for x in encode(part)]
        if len(ids) >= n_tokens or len(part) >= len(text):
            return keep(ids)
        take *= 2


def encode_window(text: str, encode, n_tokens: int, window: str = "tail") -> list[int]:
    """Token ids of one case's window of at most n_tokens tokens, tokenising only that end of the text."""
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")
    if n_tokens <= 0:
        return []
    if window == "tail":
        return _grow(text, encode, n_tokens, tail_text, lambda ids: ids[-n_tokens:])
    if window == "head":
        return _grow(text, encode, n_tokens, head_text, lambda ids: ids[:n_tokens])
    n_head = n_tokens // 4
    if len(text) <= n_tokens * 8:
        whole = [int(x) for x in encode(text)]
        if len(whole) <= n_tokens:
            return whole
        return whole[:n_head] + whole[len(whole) - (n_tokens - n_head) :]
    return encode_window(text, encode, n_head, "head") + encode_window(text, encode, n_tokens - n_head, "tail")


_ENC = None


def _worker(args):
    text, n, window = args
    return encode_window(text, _ENC, n, window)


def fingerprint(encode) -> str:
    """Short id of a tokenizer (its ids for a fixed sentence), so caches of another tokenizer are not reused."""
    s = "The appellant filed an appeal against the order of the High Court dated 2.3.2006."
    return hashlib.md5(json.dumps(list(map(int, encode(s)))).encode()).hexdigest()[:10]


def encode_docs(
    docs: list[dict],
    encode,
    n_tokens: int,
    window: str = "tail",
    workers: int = 1,
    cache_dir: str | None = None,
    log=print,
) -> list[list[int]]:
    """Window token ids for every case (same order)."""
    global _ENC
    out: list[list[int] | None] = [None] * len(docs)
    cache, path = {}, None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"tok_{fingerprint(encode)}_{window}_{n_tokens}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    cache = pickle.load(f)
            except Exception as e:  # noqa: BLE001  (a damaged cache file is rebuilt)
                log(f"[data] cache {path} unreadable ({type(e).__name__}); tokenising again")
                cache = {}
    todo = []
    for i, d in enumerate(docs):
        key = hashlib.md5(d["text"].encode("utf-8")).hexdigest()
        if key in cache:
            out[i] = cache[key]
        else:
            todo.append((i, key))
    if todo:
        log(f"[data] tokenising {len(todo)} cases ({window} {n_tokens} tokens) with {workers} process(es)...")
        jobs = [(docs[i]["text"], n_tokens, window) for i, _ in todo]
        _ENC = encode
        if workers > 1 and len(jobs) >= 4 * workers and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(workers) as pool:
                res = pool.map(_worker, jobs, chunksize=max(1, len(jobs) // (workers * 8)))
        else:
            res = [_worker(j) for j in jobs]
        for (i, key), ids in zip(todo, res):
            out[i] = ids
            cache[key] = ids
        if path:
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
    return out  # type: ignore[return-value]


def chunk_span(max_len: int, stride: int, k: int) -> int:
    """Tokens needed at the end of a case for its last k chunks of max_len tokens, consecutive chunks `stride` apart."""
    return stride * (k - 1) + max_len


def chunk_ids(ids: list[int], max_len: int, stride: int, k: int) -> list[list[int]]:
    """The last (up to) k chunks of max_len tokens, in reading order."""
    n = len(ids)
    if n <= max_len:
        return [list(ids)]
    out, end, prev_start = [], n, None
    while len(out) < k:
        start = end - max_len
        if start <= 0:
            if prev_start is None or prev_start >= stride // 2:
                out.append(list(ids[:max_len]))
            break
        out.append(list(ids[start:end]))
        prev_start = start
        end -= stride
    return out[::-1]


def length_report(lengths: list[int], n_tokens: int) -> str:
    s = sorted(lengths)
    if not s:
        return "no cases"
    full = sum(1 for x in s if x >= n_tokens)
    return (
        f"window tokens per case: median {s[len(s) // 2]}, min {s[0]}, max {s[-1]}; "
        f"{full} of {len(s)} ({full / len(s):.0%}) fill the whole {n_tokens}-token window"
    )


# --- batching
def batches(
    lengths: list[int], batch_size: int, shuffle: bool, seed: int = 0, exact: bool = False, pool: int = 50
) -> list[list[int]]:
    """Index batches of similar length."""
    idx = list(range(len(lengths)))
    rng = random.Random(seed)
    if exact:
        groups: dict[int, list[int]] = {}
        for i in idx:
            groups.setdefault(lengths[i], []).append(i)
        out = []
        for L in sorted(groups):
            g = groups[L]
            if shuffle:
                rng.shuffle(g)
            out += [g[i : i + batch_size] for i in range(0, len(g), batch_size)]
        if shuffle:
            rng.shuffle(out)
        return out
    if not shuffle:
        idx.sort(key=lambda i: lengths[i])
        return [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
    rng.shuffle(idx)
    out = []
    for c in range(0, len(idx), batch_size * pool):
        part = sorted(idx[c : c + batch_size * pool], key=lambda i: lengths[i])
        out += [part[i : i + batch_size] for i in range(0, len(part), batch_size)]
    rng.shuffle(out)
    return out


def class_weights(y: list[int], power: float = 0.5, cap: float = 10.0) -> list[float]:
    """w_c = min(cap, (count of the bigger class / count_c) ** power); [1, 1] for balanced data."""
    c = Counter(y)
    top = max(c.values())
    return [min(cap, (top / c[k]) ** power) if c.get(k) else 1.0 for k in (0, 1)]


def make_folds(docs: list[dict], k: int = 5, seed: int = 0) -> list[list[int]]:
    """k groups of indices with the same 0/1 balance."""
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for lab in (0, 1):
        idx = [i for i, d in enumerate(docs) if d["label"] == lab]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [sorted(f) for f in folds]


# --- metrics
def report(y: list[int], p1: list[float], threshold: float = 0.5) -> dict:
    """Binary scores for P(label 1) = p1 cut at `threshold`: per class precision / recall / F1, macro-F1 (Codabench's
    main metric), macro precision / recall, accuracy and the confusion counts."""
    pred = [int(p >= threshold) for p in p1]
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    per = {}
    for c, (t, f_pos, f_neg) in ((1, (tp, fp, fn)), (0, (tn, fn, fp))):
        p = t / (t + f_pos) if t + f_pos else 0.0
        r = t / (t + f_neg) if t + f_neg else 0.0
        per[c] = dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, support=t + f_neg)
    n = len(y)
    return dict(
        threshold=threshold,
        n=n,
        accuracy=(tp + tn) / n if n else 0.0,
        macro_f1=(per[0]["f1"] + per[1]["f1"]) / 2,
        macro_precision=(per[0]["precision"] + per[1]["precision"]) / 2,
        macro_recall=(per[0]["recall"] + per[1]["recall"]) / 2,
        class0=per[0],
        class1=per[1],
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        predicted_1=sum(pred),
        gold_1=sum(y),
    )


def tune_threshold(y: list[int], p1: list[float], grid=None) -> tuple[float, float]:
    """(threshold, macro-F1) with the highest macro-F1; ties go to the threshold nearest 0.5."""
    grid = grid if grid is not None else [i / 100 for i in range(2, 99)]
    best = None
    for t in grid:
        f = report(y, p1, t)["macro_f1"]
        key = (round(f, 10), -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, t, f)
    return best[1], best[2]


def format_report(rep: dict, title: str = "") -> str:
    lines = [
        f"{title}macro-F1 {rep['macro_f1']:.4f} | accuracy {rep['accuracy']:.4f} | threshold {rep['threshold']:.2f}"
    ]
    for c in (0, 1):
        r = rep[f"class{c}"]
        lines.append(
            f"    class {c}: precision {r['precision']:.3f} recall {r['recall']:.3f} "
            f"F1 {r['f1']:.3f} (n={r['support']})"
        )
    lines.append(
        f"    confusion: true 0 -> predicted 0 {rep['tn']}, predicted 1 {rep['fp']} | "
        f"true 1 -> predicted 0 {rep['fn']}, predicted 1 {rep['tp']}"
    )
    return "\n".join(lines)


# --- submission
def write_submission(test_docs: list[dict], preds: dict[str, int], out_dir: str) -> str:
    """prediction.json = [{"id": ..., "pred": 0 or 1}, ...] in test.json order, and submission.zip holding it."""
    os.makedirs(out_dir, exist_ok=True)
    want = [d["id"] for d in test_docs]
    missing = [i for i in want if i not in preds]
    extra = set(preds) - set(want)
    if missing or extra:
        raise ValueError(f"prediction ids do not match test.json: missing {missing[:3]}, extra {sorted(extra)[:3]}")
    rows = []
    for i in want:
        v = preds[i]
        if isinstance(v, bool) or int(v) not in (0, 1) or int(v) != v:
            raise ValueError(f"prediction for {i} is {v!r}, not 0 or 1")
        rows.append({"id": i, "pred": int(v)})
    path = os.path.join(out_dir, "prediction.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    zpath = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    with zipfile.ZipFile(zpath) as z:
        back = json.loads(z.read("prediction.json"))
    if len(back) != len(want) or any(type(r["pred"]) is not int for r in back):
        raise RuntimeError("submission.zip check failed")
    return zpath
