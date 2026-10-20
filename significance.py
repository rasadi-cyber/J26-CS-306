"""Paired significance tests across seeds (same seeds = same data, drones and attackers for every method).

  .venv/bin/python significance.py FINAL                     # Jiang + DQN vs every other method, all key metrics
  .venv/bin/python significance.py FINAL jiang_dqn fedavg     # one pair

Paired t-test (two-sided) on the per-seed differences of the last-10 test metrics. With 5 seeds a Wilcoxon
signed-rank test can never go below p = 0.0625, so it is shown only as a sign check.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
METRICS = [("macro_f1", True), ("rare_f1", True), ("missed_attack_rate", False), ("false_alarm_rate", False)]


def per_seed(run, method, metric):
    meta = json.load(open(os.path.join(ROOT, run, "meta.json")))
    out = {}
    for s in meta["seeds"]:
        p = os.path.join(ROOT, run, method, f"seed_{s}", "summary.json")
        if os.path.exists(p):
            t = json.load(open(p))["test_last10"]
            if metric in t:
                out[s] = t[metric]
    return out


def compare(run, a, b):
    rows = []
    for metric, higher_better in METRICS:
        x, y = per_seed(run, a, metric), per_seed(run, b, metric)
        seeds = sorted(set(x) & set(y))
        if len(seeds) < 2:
            continue
        d = np.array([x[s] - y[s] for s in seeds])
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(stats.ttest_rel([x[s] for s in seeds], [y[s] for s in seeds]).pvalue)
        better = (d.mean() > 0) == higher_better and not np.allclose(d, 0)
        wins = int(np.sum(d > 0) if higher_better else np.sum(d < 0))
        rows.append((metric, np.mean([x[s] for s in seeds]), np.mean([y[s] for s in seeds]), d.mean(), p, wins, len(seeds),
                     better))
    return rows


def main(argv):
    run = argv[0] if argv else "FINAL"
    pairs = [(argv[1], argv[2])] if len(argv) >= 3 else \
        [("jiang_dqn", m) for m in ("jiang_fixed", "fedavg", "multi_krum", "fedavg_clean", "jiang_dqn_pre")]
    for a, b in pairs:
        rows = compare(run, a, b)
        if not rows:
            continue
        print(f"\n{a} vs {b}  ({run})")
        for metric, ma, mb, d, p, wins, n, better in rows:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {metric:20s} {ma:.4f} vs {mb:.4f}  diff {d:+.4f}  p={p:.4f} {sig:4s}  "
                  f"{'better' if better else 'not better'} on {wins}/{n} seeds")


if __name__ == "__main__":
    main(sys.argv[1:])
