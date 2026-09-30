"""Byte-level BPE tokenizer from scratch (educational).

Idea:
  1. Start with 256 base tokens = all possible bytes (so ANY text is encodable, no <unk>).
  2. Repeatedly find the most frequent adjacent pair of tokens in the training data,
     merge it into a NEW token, and record that merge rule.
  3. Encoding = split text into bytes, then re-apply the merge rules in the order learned.
"""
import json
import re
from collections import Counter

# Pre-tokenizer: split text into "words" first, so merges never cross word boundaries.
# A leading space stays attached to the word (" Court"), like GPT-2.
PRETOKEN_RE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


class BPETokenizer:
    def __init__(self, special_tokens=("<pad>", "<unk>", "<bos>", "<eos>")):
        self.merges = {}                                   # (id_a, id_b) -> new_id, in learn order
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.special_tokens = list(special_tokens)
        self.special_ids = {}

    # ------------------------------------------------------------------ training
    def train(self, texts, vocab_size, min_frequency=2, verbose=True):
        # 1) Count words once. Each word = tuple of byte ids. Training works on UNIQUE
        #    words weighted by frequency -> much faster than scanning raw text.
        word_freq = Counter()
        for text in texts:
            for w in PRETOKEN_RE.findall(text):
                word_freq[tuple(w.encode("utf-8"))] += 1
        words = {w: list(w) for w in word_freq}           # current tokenization of each word

        n_merges = vocab_size - 256 - len(self.special_tokens)
        for step in range(n_merges):
            # 2) Count every adjacent pair across all words (weighted by word frequency).
            pair_counts = Counter()
            for w, toks in words.items():
                f = word_freq[w]
                for a, b in zip(toks, toks[1:]):
                    pair_counts[(a, b)] += f
            if not pair_counts:
                break
            best, count = pair_counts.most_common(1)[0]
            if count < min_frequency:
                break

            # 3) Create the new token and replace every occurrence of the pair.
            new_id = 256 + len(self.merges)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            for w, toks in words.items():
                words[w] = self._merge(toks, best, new_id)

            if verbose and step % 200 == 0:
                print(f"merge {step}: {self.vocab[new_id]!r}  (count={count})")

        # 4) Special tokens get the last ids; they are never produced by merges.
        for tok in self.special_tokens:
            self.special_ids[tok] = len(self.vocab)
            self.vocab[len(self.vocab)] = tok.encode("utf-8")

    @staticmethod
    def _merge(toks, pair, new_id):
        """Replace every non-overlapping occurrence of `pair` in toks with new_id."""
        out, i = [], 0
        while i < len(toks):
            if i < len(toks) - 1 and toks[i] == pair[0] and toks[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(toks[i])
                i += 1
        return out

    # ------------------------------------------------------------------ encode / decode
    def _encode_word(self, word):
        toks = list(word.encode("utf-8"))
        while len(toks) >= 2:
            # Apply the EARLIEST-learned merge available (lowest new_id) -> same order as training.
            pairs = set(zip(toks, toks[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            toks = self._merge(toks, pair, self.merges[pair])
        return toks

    def encode(self, text):
        ids = []
        for w in PRETOKEN_RE.findall(text):
            ids.extend(self._encode_word(w))
        return ids

    def decode(self, ids):
        # Join raw bytes, then decode; errors="replace" handles ids that cut a UTF-8 char.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_to_id(self, tok):
        return self.special_ids.get(tok)

    def get_vocab_size(self):
        return len(self.vocab)

    # ------------------------------------------------------------------ save / load
    def save(self, path):
        data = {"merges": [[a, b, n] for (a, b), n in self.merges.items()],
                "special_tokens": self.special_tokens}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls(data["special_tokens"])
        for a, b, n in data["merges"]:                  # rebuild vocab in learn order
            tok.merges[(a, b)] = n
            tok.vocab[n] = tok.vocab[a] + tok.vocab[b]
        for s in tok.special_tokens:
            tok.special_ids[s] = len(tok.vocab)
            tok.vocab[len(tok.vocab)] = s.encode("utf-8")
        return tok


if __name__ == "__main__":
    import corpus
    docs = list(corpus.iter_docs("supreme_court", max_docs=50))   # small sample: pure Python is slow
    tok = BPETokenizer()
    tok.train(docs, vocab_size=2000)
    s = "The appellant filed a Special Leave Petition under Article 136 of the Constitution."
    ids = tok.encode(s)
    print(len(s.split()), "words ->", len(ids), "tokens")
    print([tok.vocab[i] for i in ids])
    assert tok.decode(ids) == s                                    # round-trip check