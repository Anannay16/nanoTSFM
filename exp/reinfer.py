"""Copy a trained run with inference-only options switched on, so it can be re-scored without
retraining. The weights are untouched; only the checkpoint's model_config changes.

Usage: uv run python exp/reinfer.py runs/rec-s7 runs/rec-s7-flip flip_symmetric
       uv run python exp/reinfer.py runs/rec-s7 runs/rec-s7-both flip_symmetric clamp_positive
Then:  ./run.sh eval rec-s7-flip
"""

import shutil
import sys
from pathlib import Path

import torch

source, target, *flags = sys.argv[1:]
source, target = Path(source), Path(target)
if not flags:
    sys.exit(__doc__)
if target.exists():
    sys.exit(f"{target} already exists")
target.mkdir(parents=True)
for name in ("gep-val.json", "run.json", "config.json", "training.jsonl"):
    if (source / name).exists():
        shutil.copy2(source / name, target / name)
checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
for flag in flags:
    if flag not in ("flip_symmetric", "clamp_positive"):
        sys.exit(f"Unknown inference option {flag}")
    checkpoint["model_config"][flag] = True
torch.save(checkpoint, target / "checkpoint.pt")
print(f"{target}: {checkpoint['model_config']}")
