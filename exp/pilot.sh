#!/usr/bin/env bash
# Tonight's runs: reproduce Shu's record (3 seeds) + one-seed pilots.
# Trains run one after another on the GPU; each finished run is scored on GIFT-Eval in the
# background while the next one trains. Usage, from /workspace/nanoTSFM:
#   nohup bash exp/pilot.sh > exp/pilot.log 2>&1 &
#   tail -f exp/pilot.log            # watch;  python3 exp/compare.py   # results so far
set -uo pipefail
cd "$(dirname "$0")/.."

RUNS_TO_DO=(
  "rec-s7 configs/series.yaml 7"
  "freq05-s7 configs/freq05.yaml 7"
  "freq1-s7 configs/freq1.yaml 7"
  "power15-s7 configs/power15.yaml 7"
  "rec-s1 configs/series.yaml 1"
  "rec-s2 configs/series.yaml 2"
)
[[ $# -gt 0 ]] && RUNS_TO_DO=("$@")   # or pass your own: bash exp/pilot.sh "name config seed" ...

eval_pid=""
for spec in "${RUNS_TO_DO[@]}"; do
  read -r name config seed <<<"$spec"
  if [[ -f runs/$name/gift.json ]]; then echo ">>> $name already done, skipping"; continue; fi
  if [[ ! -f runs/$name/checkpoint.pt ]]; then
    rm -rf "runs/$name"
    echo ">>> $(date +%T) train $name ($config, seed $seed)"
    ./run.sh train "$name" "$config" "$seed" || { echo "!!! train $name FAILED"; continue; }
  fi
  [[ -n $eval_pid ]] && wait "$eval_pid"     # one GIFT-Eval at a time (it uses every CPU core)
  echo ">>> $(date +%T) eval $name (background)"
  ( ./run.sh eval "$name" > "runs/$name/eval.log" 2>&1 \
      && echo ">>> $(date +%T) done $name: $(tail -n 1 runs/$name/eval.log)" \
      || echo "!!! eval $name FAILED, see runs/$name/eval.log" ) &
  eval_pid=$!
done
[[ -n $eval_pid ]] && wait "$eval_pid"
echo ">>> $(date +%T) ALL DONE"
python3 exp/compare.py
