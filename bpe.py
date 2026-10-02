"""Byte-level BPE tokenizer implemented from scratch (no tokenizer library).

Algorithm (same idea as GPT-2 / course BPE):
  1. Base vocab = 256 single bytes  -> any text is encodable, no <unk> needed.
  2. Pre-tokenize text into "words" (regex), so merges never cross word boundaries.
  3. Training: repeatedly merge the most frequent adjacent token pair into a new token.
  4. Encoding: split a word into bytes, re-apply merges in the order they were learned.

Speed tricks (pure Python, but usable on tens of MB):
  * train on UNIQUE words weighted by frequency, not raw text
  * incremental pair counts: after a merge, only words containing that pair are updated
  * max-heap with lazy deletion to find the most frequent pair quickly
  * encode cache: each distinct word is BPE-encoded once (legal text is very repetitive)
"""
import heapq
import json
import re
from collections import Counter, defaultdict

# " ?word" | " ?number" | " ?punctuation-run" | trailing spaces | other whitespace
PRETOKEN_RE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
DEFAULT_SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>")


def _merge(toks, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in toks with new_id."""
    a, b = pair
    out, i, n = [], 0, len(toks)
    while i < n:
        if i < n - 1 and toks[i] == a and toks[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, special_tokens=DEFAULT_SPECIALS):
        self.special_tokens = list(special_tokens)
        self.merges = {}                                   # (a, b) -> new_id  (= merge rank)
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.special_ids = {}
        self._cache = {}

    # ------------------------------------------------------------------ training
    def train(self, texts, vocab_size, min_frequency=2, verbose=True):
        # 1) unique words with counts; each word starts as a list of byte ids
        word_freq = Counter()
        for t in texts:
            word_freq.update(PRETOKEN_RE.findall(t))
        words = [list(w.encode("utf-8")) for w in word_freq]
        freqs = list(word_freq.values())
        if verbose:
            print(f"[bpe] {sum(freqs):,} words, {len(words):,} unique", flush=True)

        # 2) initial pair counts + index pair -> words containing it
        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for idx, (toks, f) in enumerate(zip(words, freqs)):
            for p in zip(toks, toks[1:]):
                pair_counts[p] += f
                where[p].add(idx)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        n_merges = vocab_size - 256 - len(self.special_tokens)
        for step in range(n_merges):
            # 3) pop the most frequent pair; skip stale heap entries (lazy deletion)
            best = None
            while heap:
                neg_c, p = heapq.heappop(heap)
                if pair_counts.get(p, 0) == -neg_c and -neg_c > 0:
                    best, count = p, -neg_c
                    break
            if best is None or count < min_frequency:
                break

            new_id = 256 + len(self.merges)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            # 4) update only the words that contain `best`
            changed = set()
            for idx in where.pop(best, ()):
                old = words[idx]
                new = _merge(old, best, new_id)
                if len(new) == len(old):          # stale index entry, pair no longer here
                    continue
                f = freqs[idx]
                for p in zip(old, old[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                for p in zip(new, new[1:]):
                    pair_counts[p] += f
                    where[p].add(idx)
                    changed.add(p)
                words[idx] = new
            pair_counts.pop(best, None)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)

            if verbose and step % 1000 == 0:
                print(f"[bpe] merge {step:6d}: {self.vocab[new_id]!r} (count={count})", flush=True)

        self._add_specials()
        if verbose:
            print(f"[bpe] done: vocab={self.get_vocab_size()}", flush=True)

    def _add_specials(self):
        self.special_ids = {}
        for s in self.special_tokens:
            self.special_ids[s] = len(self.vocab)
            self.vocab[len(self.vocab)] = s.encode("utf-8")

    # ------------------------------------------------------------------ encode / decode
    def _encode_word(self, word):
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        toks = list(word.encode("utf-8"))
        while len(toks) >= 2:
            # earliest-learned merge present in the word (lowest rank) is applied first
            best, best_rank = None, None
            for p in zip(toks, toks[1:]):
                r = self.merges.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            toks = _merge(toks, best, best_rank)
        if len(self._cache) < 2_000_000:          # bound memory
            self._cache[word] = toks
        return toks

    def encode(self, text):
        ids = []
        for w in PRETOKEN_RE.findall(text):
            ids.extend(self._encode_word(w))
        return ids

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_to_id(self, tok):
        return self.special_ids.get(tok)

    def get_vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ------------------------------------------------------------------ save / load
    def save(self, path):
        data = {"type": "byte_level_bpe_scratch",
                "pattern": PRETOKEN_RE.pattern,
                "special_tokens": self.special_tokens,
                "merges": [[a, b] for (a, b) in self.merges]}   # in learn order
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls(data["special_tokens"])
        for a, b in data["merges"]:
            new_id = 256 + len(tok.merges)
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok._add_specials()
        return tok