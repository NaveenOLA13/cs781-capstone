"""Item 1: tokenizer adapters that return clean character offsets.

Every adapter returns (token_ids, offsets) for the CONTENT tokens only (no special
tokens). Special tokens are added per window in `windows.py`, so window logic is the
same for every model.

Offsets are normalised to the ORIGINAL text: leading/trailing whitespace inside a
token's offset (SentencePiece '▁' tokens) is stripped, and whitespace-only tokens get
(-1, -1) so they never carry a label.
"""
from __future__ import annotations

import re
from typing import Protocol


class TokenizerAdapter(Protocol):
    cls_id: int | None
    sep_id: int | None
    pad_id: int

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]: ...


def normalize_offsets(text: str, offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for s, e in offsets:
        if s is None or e is None or e <= s:
            out.append((-1, -1))
            continue
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        out.append((s, e) if e > s else (-1, -1))
    return out


class HFAdapter:
    """Wraps a Hugging Face *fast* tokenizer (AutoTokenizer.from_pretrained(..., use_fast=True))."""

    def __init__(self, hf_tokenizer):
        if not getattr(hf_tokenizer, "is_fast", False):
            raise ValueError("Need a fast tokenizer for offset mapping.")
        self.tok = hf_tokenizer
        self.cls_id = hf_tokenizer.cls_token_id if hf_tokenizer.cls_token_id is not None else hf_tokenizer.bos_token_id
        self.sep_id = hf_tokenizer.sep_token_id if hf_tokenizer.sep_token_id is not None else hf_tokenizer.eos_token_id
        self.pad_id = hf_tokenizer.pad_token_id or 0

    def encode(self, text: str):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), normalize_offsets(text, [tuple(o) for o in enc["offset_mapping"]])


class ToySentencePieceAdapter:
    """Test tokenizer that reproduces SentencePiece quirks: the leading space is part of
    the first piece's offset, words are split into subwords of <= `piece` chars, and
    punctuation is its own token. Used for tests and the round-trip check only."""

    _re = re.compile(r"\s*(\w+|[^\w\s])")

    def __init__(self, piece: int = 3):
        self.piece = piece
        self.cls_id, self.sep_id, self.pad_id = 1, 2, 0
        self._vocab: dict[str, int] = {}

    def _id(self, s: str) -> int:
        return self._vocab.setdefault(s, len(self._vocab) + 3)

    def encode(self, text: str):
        ids, offs = [], []
        for m in self._re.finditer(text):
            ws_start, w_start, w_end = m.start(), m.start(1), m.end(1)
            for i, p in enumerate(range(w_start, w_end, self.piece)):
                s = ws_start if i == 0 else p  # first piece swallows the leading space
                e = min(p + self.piece, w_end)
                ids.append(self._id(text[s:e]))
                offs.append((s, e))
        return ids, normalize_offsets(text, offs)


# ---- your own BPE ------------------------------------------------------------------
_MARKERS = ("Ġ", "▁", "Ċ", "</w>", "##", "@@")


def default_clean(piece: str) -> str:
    """Remove common BPE word-boundary markers. Pass your own if your BPE differs."""
    for m in _MARKERS:
        piece = piece.replace(m, "")
    return piece


def align_offsets(text: str, pieces: list[str], clean=default_clean, lookahead: int = 50):
    """Recover character offsets for ANY tokenizer that can give you each token's string.
    Walks a cursor through `text`, skipping whitespace, and matches each cleaned piece.
    Returns (offsets, n_failed). A failed piece gets (-1,-1) and the cursor doesn't move.
    n_failed must be 0 on your data; otherwise fix `clean` (e.g. byte-level BPE needs
    bytes decoded back to text first)."""
    offs, cur, failed = [], 0, 0
    for p in pieces:
        t = clean(p)
        if not t.strip():
            offs.append((-1, -1))
            continue
        t = t.strip()
        k = cur
        while k < len(text) and text[k].isspace():
            k += 1
        if text.startswith(t, k):
            offs.append((k, k + len(t)))
            cur = k + len(t)
            continue
        j = text.find(t, cur, cur + lookahead + len(t))
        if j >= 0:
            offs.append((j, j + len(t)))
            cur = j + len(t)
        else:
            offs.append((-1, -1))
            failed += 1
    return offs, failed


class AlignedAdapter:
    """Wraps YOUR existing BPE without changing it.
        encode_fn(text) -> list[int]           token ids
        piece_fn(ids)   -> list[str]           each token's string (e.g. [vocab_inv[i] for i in ids])
    Raises if any token can't be aligned, so offset bugs can't pass silently."""

    def __init__(self, encode_fn, piece_fn, cls_id=None, sep_id=None, pad_id=0,
                 clean=default_clean, strict=True):
        self.encode_fn, self.piece_fn, self.clean, self.strict = encode_fn, piece_fn, clean, strict
        self.cls_id, self.sep_id, self.pad_id = cls_id, sep_id, pad_id
        self.total_failed = 0

    def encode(self, text: str):
        ids = list(self.encode_fn(text))
        offs, failed = align_offsets(text, self.piece_fn(ids), self.clean)
        self.total_failed += failed
        if failed and self.strict:
            bad = [p for p, o in zip(self.piece_fn(ids), offs) if o[0] < 0][:10]
            raise ValueError(f"{failed} tokens could not be aligned, e.g. {bad!r}; fix `clean`")
        return ids, offs


class OffsetBPE:
    """Minimal character-level BPE that tracks offsets natively. Use it only if your own
    BPE can't produce token strings. Pre-tokenises into words/punctuation, then merges
    inside each word."""

    _pre = re.compile(r"\w+|[^\w\s]")

    def __init__(self):
        self.merges: dict[tuple[str, str], int] = {}
        self.vocab: dict[str, int] = {"<pad>": 0, "<cls>": 1, "<sep>": 2, "<unk>": 3}
        self.cls_id, self.sep_id, self.pad_id = 1, 2, 0

    def train(self, texts, vocab_size: int = 8000, min_freq: int = 2):
        from collections import Counter
        words = Counter(m.group() for t in texts for m in self._pre.finditer(t))
        splits = {w: list(w) for w in words}
        for w in words:
            for ch in w:
                self.vocab.setdefault(ch, len(self.vocab))
        while len(self.vocab) < vocab_size:
            pairs = Counter()
            for w, f in words.items():
                s = splits[w]
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            (a, b), f = pairs.most_common(1)[0]
            if f < min_freq:
                break
            self.merges[(a, b)] = len(self.merges)
            self.vocab.setdefault(a + b, len(self.vocab))
            for w in words:
                s, i, out = splits[w], 0, []
                while i < len(s):
                    if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                        out.append(a + b); i += 2
                    else:
                        out.append(s[i]); i += 1
                splits[w] = out
        return self

    def _bpe(self, word: str) -> list[str]:
        s = list(word)
        while len(s) > 1:
            ranked = [(self.merges.get((x, y), 10**9), i) for i, (x, y) in enumerate(zip(s, s[1:]))]
            r, i = min(ranked)
            if r == 10**9:
                break
            s[i:i + 2] = [s[i] + s[i + 1]]
        return s

    def encode(self, text: str):
        ids, offs = [], []
        for m in self._pre.finditer(text):
            pos = m.start()
            for p in self._bpe(m.group()):
                ids.append(self.vocab.get(p, 3))
                offs.append((pos, pos + len(p)))
                pos += len(p)
        return ids, offs