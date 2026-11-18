#!/usr/bin/env bash
# Final submission runs: three seeds of configs/final.yaml at one clean commit, scored on
# GIFT-Eval, then packaged as records/<name>/result.json with ./run.sh submit.
#   bash exp/final.sh 2026-11-19_Anannay-Sharma
set -uo pipefail
cd "$(dirname "$0")/.."
name=${1:?Usage: bash exp/final.sh <YYYY-MM-DD>_<name>}
training=(src configs scripts run.sh pyproject.toml uv.lock)
if [[ -n $(git status --porcelain -- "${training[@]}") ]]; then
  echo "Training files are modified or untracked; commit them first:"
  git status --short -- "${training[@]}"
  exit 1
fi
echo ">>> $(date +%T) final runs at commit $(git rev-parse --short HEAD)"
bash exp/pilot.sh "final-s7 configs/final.yaml 7" "final-s1 configs/final.yaml 1" \
  "final-s2 configs/final.yaml 2"
./run.sh submit "$name" final-s7 final-s1 final-s2
echo ">>> $(date +%T) FINAL DONE: records/$name/result.json"
