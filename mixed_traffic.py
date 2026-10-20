"""Mixed-traffic test (separate evaluation, no retraining): normal traffic with attack bursts interleaved in time.

  .venv/bin/python mixed_traffic.py            # evaluates runs/FINAL, writes runs/FINAL/mixed_traffic.json
  .venv/bin/python mixed_traffic.py RUN_NAME
"""
import json
import os
import sys

from fldos.mixed import evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "FINAL"
    run_dir = os.path.join(ROOT, "runs", run)
    cfg = json.load(open(os.path.join(run_dir, "meta.json")))["config"]
    out = evaluate(cfg, ROOT, run_dir)
    print("rows per class:", out["episodes"]["rows_per_class"])
    for m, e in out["methods"].items():
        print(f"{m:14s} macro {e['macro_f1']['mean']:.3f} acc {e['accuracy']['mean']:.3f} De-auth {e['rare_f1']['mean']:.2f} "
              f"missed {e['missed_attack_rate']['mean']:.3f} false alarms {e['false_alarm_rate']['mean']:.3f}")
