"""Combine trained runs of one configuration into an ensemble run whose forecast averages their
quantiles. A cheap test of ensembling before building it into training.

Usage: uv run python exp/ensemble.py runs/rec-ens3 runs/rec-s7 runs/rec-s1 runs/rec-s2
Then:  ./run.sh eval rec-ens3
"""

import sys
from pathlib import Path

import torch

target, *sources = map(Path, sys.argv[1:])
if len(sources) < 2:
    sys.exit(__doc__)
if target.exists():
    sys.exit(f"{target} already exists")
checkpoints = [torch.load(s / "checkpoint.pt", map_location="cpu", weights_only=True) for s in sources]
if any(c["model_config"] != checkpoints[0]["model_config"] for c in checkpoints):
    sys.exit("All runs must share one model configuration")
combined = dict(checkpoints[0])
combined["members"] = [c["model"] for c in checkpoints]
combined["elapsed_seconds"] = sum(c["elapsed_seconds"] for c in checkpoints)
target.mkdir(parents=True)
torch.save(combined, target / "checkpoint.pt")
print(f"{target}: {len(sources)} members, {combined['elapsed_seconds']:.0f} s of training in all")
