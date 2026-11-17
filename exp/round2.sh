#!/usr/bin/env bash
# Round 2: one-seed pilots of each new switch, each a single change from Shu's record
# (configs/series.yaml), plus free inference-only re-scores of round 1's rec-s7.
# Needs runs/rec-s7 from round 1. Usage, from /workspace/nanoTSFM:
#   nohup bash exp/round2.sh > exp/round2.log 2>&1 &
#   tail -f exp/round2.log          # then: python3 exp/compare.py
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f runs/rec-s7/checkpoint.pt ]] || { echo "Run round 1 first (needs runs/rec-s7)"; exit 1; }

# Inference-only variants: same weights, new forecast() options. No training needed.
for spec in "flip flip_symmetric" "pos clamp_positive" "flippos flip_symmetric clamp_positive"; do
  read -r tag flags <<<"$spec"
  [[ -d runs/rec-s7-$tag ]] || uv run --locked --extra gift ${TORCH:+--extra $TORCH} python exp/reinfer.py runs/rec-s7 "runs/rec-s7-$tag" $flags
done

# Ensemble of round 1's three record seeds: is averaging models worth building into training?
[[ -d runs/rec-ens3-s7 ]] || uv run --locked --extra gift ${TORCH:+--extra $TORCH} python exp/ensemble.py runs/rec-ens3-s7 \
  runs/rec-s7 runs/rec-s1 runs/rec-s2

# Interleave so the GPU trains while the CPUs score. Inference-only runs already have a
# checkpoint, so pilot.sh skips straight to scoring them.
bash exp/pilot.sh \
  "span16-s7 configs/span16.yaml 7" \
  "rec-s7-flip - 7" \
  "ctx2048-s7 configs/ctx2048.yaml 7" \
  "rec-s7-pos - 7" \
  "ema-s7 configs/ema.yaml 7" \
  "rec-s7-flippos - 7" \
  "wd01-s7 configs/wd01.yaml 7" \
  "aug-s7 configs/aug.yaml 7" \
  "rec-ens3-s7 - 7" \
  "patch16-s7 configs/patch16.yaml 7"
