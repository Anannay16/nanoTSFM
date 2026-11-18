"""Share of training windows per frequency group under freq_power q (source_power 1).

Usage: uv run --locked --extra gift python exp/shares.py
"""

from collections import Counter

from nanotsfm.data import Windows, load

ds = load("GEP-M", "train")
rows = {}
for q in (1e-12, 0.5, 1.0):
    w = Windows(ds, 1023, 1, 1, 0, 1.0, q)
    share = Counter()
    for g, x in zip(w.groups, w.weights, strict=True):
        share[g] += x
    rows[q] = share
n = Counter(w.groups)
print(f"{'group':<9}{'sources':>8}{'q=0':>8}{'q=0.5':>8}{'q=1':>8}")
for g in sorted(n, key=lambda g: -rows[1e-12][g]):
    print(f"{g:<9}{n[g]:>8}" + "".join(f"{rows[q][g] * 100:>7.1f}%" for q in rows))
