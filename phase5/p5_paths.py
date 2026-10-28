"""Makes the Phase 4 building blocks importable from phase5/: the fine-tuning techniques (cj_tech), model loading
(cj_models, which itself adds phase3/ and the repository root), data and tokenising helpers (cj_data), training
helpers (cj_train) and mid-training (cj_mid). phase5/ comes first on the path, so Phase 5 files always win.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE4 = os.path.join(os.path.dirname(HERE), "phase4")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.isdir(PHASE4) and PHASE4 not in sys.path:
    sys.path.append(PHASE4)
