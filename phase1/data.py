"""Random training windows from the uint16 token files written by prepare_data.py."""

import json
import os

import numpy as np
import torch


class TokenData:
    """Samples (x, y) windows of block_size tokens; each row picks a source with probability given by weights."""

    def __init__(self, data_dir, block_size, weights, seed=0):
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.vocab_size = json.load(f)["vocab_size"]
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        self.arrays = {"train": {}, "val": {}}
        for src in weights:
            for split in ("train", "val"):
                path = os.path.join(data_dir, f"{src}_{split}.bin")
                if os.path.exists(path) and os.path.getsize(path) > 2 * (block_size + 1):
                    self.arrays[split][src] = np.memmap(path, dtype=np.uint16, mode="r")
        self.weights = {}
        for split in ("train", "val"):
            names = list(self.arrays[split])
            w = np.array([weights[s] for s in names], dtype=np.float64)
            self.weights[split] = (names, w / w.sum())

    def get_batch(self, split, batch_size, device):
        names, p = self.weights[split]
        T = self.block_size
        x = np.empty((batch_size, T), dtype=np.int64)
        y = np.empty((batch_size, T), dtype=np.int64)
        for b, k in enumerate(self.rng.choice(len(names), size=batch_size, p=p)):
            arr = self.arrays[split][names[k]]
            i = int(self.rng.integers(0, len(arr) - T - 1))
            chunk = np.asarray(arr[i : i + T + 1], dtype=np.int64)
            x[b], y[b] = chunk[:-1], chunk[1:]
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if str(device).startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def describe(self):
        names, p = self.weights["train"]
        return ", ".join(f"{s} {w:.2f} ({len(self.arrays['train'][s]) / 1e6:.0f}M tokens)" for s, w in zip(names, p))
