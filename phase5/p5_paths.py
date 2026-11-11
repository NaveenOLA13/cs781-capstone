"""Makes the Phase 4 and Phase 3 code (and, through cj_models, phase1/) importable from phase5/."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE4 = os.path.join(os.path.dirname(HERE), "phase4")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.isdir(PHASE4) and PHASE4 not in sys.path:
    sys.path.append(PHASE4)
