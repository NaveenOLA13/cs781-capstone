"""Makes the Phase 4 building blocks importable from phase7/ (fine-tuning techniques cj_tech, model loading cj_models
(which adds phase3/ and the repository root), training helpers cj_train, mid-training cj_mid).
phase7/ comes first on the path, so Phase 7 files always win."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for name in ("phase4",):
    p = os.path.join(os.path.dirname(HERE), name)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)
