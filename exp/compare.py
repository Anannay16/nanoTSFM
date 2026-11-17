"""Print GIFT-Eval results for every scored run, grouped by config, plus a per-frequency breakdown.

Usage: python3 exp/compare.py [runs_dir]
       python3 exp/compare.py runs --worst rec-s7     # that run's 15 worst tasks
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

FREQ_BY_NAME = {
    "bizitobs_application": "second", "bizitobs_service": "second", "car_parts_with_missing": "month",
    "covid_deaths": "day", "hospital": "month", "restaurant": "day",
    "temperature_rain_with_missing": "day", "m4_yearly": "year", "m4_quarterly": "quarter",
    "m4_monthly": "month", "m4_weekly": "week", "m4_daily": "day", "m4_hourly": "hour",
}
SUFFIX = {"S": "second", "T": "minute", "H": "hour", "D": "day", "W": "week", "M": "month"}
ORDER = ["short", "medium", "long", "second", "minute", "hour", "day", "week", "month", "quarter", "year"]


def freq(dataset):
    if "/" in dataset:
        return SUFFIX[dataset.split("/")[1].lstrip("0123456789")[0]]
    return FREQ_BY_NAME[dataset]


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


args = sys.argv[1:]
worst = args[args.index("--worst") + 1] if "--worst" in args else None
args = [a for a in args if a not in ("--worst", worst)]
runs = Path(args[0] if args else "runs")
if worst:
    rows = json.loads((runs / worst / "gift.json").read_text())["rows"]
    rel = sorted(
        ((r["metrics"]["mean_weighted_sum_quantile_loss"] / r["seasonal_naive"]["mean_weighted_sum_quantile_loss"],
          r["dataset"], r["term"]) for r in rows), reverse=True)
    print(f"Worst tasks for {worst} (relative CRPS; > 1 is worse than Seasonal Naive):")
    for score, dataset, term in rel[:15]:
        print(f"  {score:7.3f}  {dataset}/{term}")
    sys.exit()
by_config = defaultdict(list)
for path in sorted(runs.glob("*/gift.json")):
    result = json.loads(path.read_text())
    # "rec-s7" -> "rec"; "rec-s7-flip" -> "rec+flip" (seed dropped, inference variant kept)
    match = re.fullmatch(r"(.*)-s\d+(-(.*))?", path.parent.name)
    config = (match[1] + (f"+{match[3]}" if match[3] else "")) if match else path.parent.name
    groups = defaultdict(list)
    for row in result["rows"]:
        rel = row["metrics"]["mean_weighted_sum_quantile_loss"] / row["seasonal_naive"]["mean_weighted_sum_quantile_loss"]
        groups[row["term"]].append(rel)
        groups[freq(row["dataset"])].append(rel)
    gep = path.parent / "gep-val.json"
    gep_crps = json.loads(gep.read_text())["summary"]["geometric_relative_crps"] if gep.exists() else float("nan")
    by_config[config].append({
        "run": path.parent.name,
        "crps": result["summary"]["geometric_relative_crps"],
        "mase": result["summary"]["geometric_relative_mase"],
        "gep": gep_crps,
        "groups": {g: geo(v) for g, v in groups.items()},
    })

if not by_config:
    sys.exit(f"No scored runs (gift.json) under {runs}/ yet")

print(f"{'run':<16}{'GIFT CRPS':>10}{'MASE':>8}{'GEP-Val':>9}")
for config, items in by_config.items():
    for r in items:
        print(f"{r['run']:<16}{r['crps']:>10.4f}{r['mase']:>8.4f}{r['gep']:>9.4f}")
print("\nMean over seeds (course point: verified mean < 0.6316; new record: <= 0.6186)")
for config, items in by_config.items():
    crps = [r["crps"] for r in items]
    mean = sum(crps) / len(crps)
    sd = (sum((c - mean) ** 2 for c in crps) / (len(crps) - 1)) ** 0.5 if len(crps) > 1 else float("nan")
    print(f"  {config:<12} n={len(crps)}  CRPS {mean:.4f} +/- {sd:.4f}")
print("\nGIFT-Eval CRPS by term and frequency (mean over seeds):")
print(f"{'':<10}" + "".join(f"{c[:11]:>12}" for c in by_config))
for group in ORDER:
    cells = []
    for items in by_config.values():
        vals = [r["groups"][group] for r in items if group in r["groups"]]
        cells.append(f"{sum(vals) / len(vals):>12.3f}" if vals else f"{'-':>12}")
    print(f"{group:<10}" + "".join(cells))
