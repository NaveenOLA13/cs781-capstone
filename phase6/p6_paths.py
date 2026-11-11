"""Makes the Phase 4 and Phase 3 code (and, through cj_models, phase1/) importable from phase6/."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for name in ("phase5", "phase4"):
    p = os.path.join(os.path.dirname(HERE), name)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)
