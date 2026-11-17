#!/usr/bin/env bash
# Round 3: does augmentation stack on power 1.5, with flip + clamp on top?
# Builds the ablation ladder: power15 -> +aug -> +flip -> +clamp (3 seeds each),
# then (optional, last) +aug+clamp without flip, to show each part's gain.
#   nohup bash -c 'bash exp/round3.sh > exp/round3.log 2>&1; runpodctl stop pod $RUNPOD_POD_ID' >/dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
UV="uv run --locked --extra gift ${TORCH:+--extra $TORCH} python"

# power 1.5 + sign flip 0.5 + downsample 0.3
sed "s/source_power: 1$/source_power: 1.5/" configs/aug.yaml > configs/aug15.yaml
grep -q "source_power: 1.5" configs/aug15.yaml || { echo "aug15.yaml not built"; exit 1; }

# 1. Train power15+aug, 3 seeds (each scored while the next trains)
bash exp/pilot.sh "aug15-s7 configs/aug15.yaml 7" "aug15-s1 configs/aug15.yaml 1" \
  "aug15-s2 configs/aug15.yaml 2"

# 2. Re-score the same 3 models with the forecast options (no retraining)
for s in 7 1 2; do
  [[ -f runs/aug15-s$s/checkpoint.pt ]] || continue
  for v in "flippos flip_symmetric clamp_positive" "flip flip_symmetric" "pos clamp_positive"; do
    read -r tag flags <<<"$v"
    [[ -d runs/aug15-s$s-$tag ]] || $UV exp/reinfer.py runs/aug15-s$s runs/aug15-s$s-$tag $flags
  done
done
# Most important first: the final candidate, then the parts for the ablation
bash exp/pilot.sh "aug15-s7-flippos - 7" "aug15-s1-flippos - 1" "aug15-s2-flippos - 2" \
  "aug15-s7-flip - 7" "aug15-s1-flip - 1" "aug15-s2-flip - 2" \
  "aug15-s7-pos - 7" "aug15-s1-pos - 1" "aug15-s2-pos - 2"
echo ">>> $(date +%T) ROUND 3 DONE"
