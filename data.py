"""Random-window batches from the token files written by prepare_data.py.

* np.memmap: the .bin files are never loaded into RAM, only the slices we read.
* Each sample: pick a SOURCE by mixing weight, then a random start position in it,
  take block_size+1 tokens -> x = first block_size, y = same shifted by one.
* The mix is just a dict of weights (set in the config) -> no re-tokenizing to change it.
"""
import json
import os

import numpy as np
import torch


class TokenData:
    def __init__(self, data_dir, block_size, weights, seed=0):
        meta = json.load(open(os.path.join(data_dir, "meta.json")))
        self.vocab_size = meta["vocab_size"]
        self.eos_id = meta["eos_id"]
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        self.splits = {}
        for split in ("train", "val"):
            arrays = {}
            for src in weights:
                path = os.path.join(data_dir, f"{src}_{split}.bin")
                if os.path.exists(path) and os.path.getsize(path) > 2 * (block_size + 1):
                    arrays[src] = np.memmap(path, dtype=np.uint16, mode="r")
            names = [s for s in arrays if weights[s] > 0]
            if not names:
                raise ValueError(f"no usable {split} data for weights {weights}")
            p = np.array([weights[s] for s in names], dtype=np.float64)
            self.splits[split] = (names, arrays, p / p.sum())

    def describe(self):
        names, arrays, p = self.splits["train"]
        return {s: {"weight": round(float(w), 4), "train_tokens": int(len(arrays[s]))}
                for s, w in zip(names, p)}

    def get_batch(self, split, batch_size, device):
        names, arrays, p = self.splits[split]
        which = self.rng.choice(len(names), size=batch_size, p=p)
        T = self.block_size
        x = np.empty((batch_size, T), dtype=np.int64)
        y = np.empty((batch_size, T), dtype=np.int64)
        for i, k in enumerate(which):
            arr = arrays[names[k]]
            s = int(self.rng.integers(0, len(arr) - T - 1))
            chunk = arr[s:s + T + 1].astype(np.int64)
            x[i], y[i] = chunk[:-1], chunk[1:]
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
