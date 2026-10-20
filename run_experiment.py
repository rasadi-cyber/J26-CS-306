"""CLI.

  python run_experiment.py                                  # all methods x all seeds in configs/default.json
  python run_experiment.py --methods fedavg jiang_dqn --seeds 100 101 --rounds 100 --name tune-1
  python run_experiment.py --set clients.num_poisoned=0      # override any config value
  python run_experiment.py --pretrain-dqn --seeds 100 101 102   # build the carried-over DQN (tuning seeds only)
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fldos.runner import pretrain_dqn, run  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    try:
        node[keys[-1]] = json.loads(value)
    except json.JSONDecodeError:  # plain text value, e.g. data.split=temporal
        node[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.json"))
    ap.add_argument("--name", help="run folder name under runs/")
    ap.add_argument("--out", help="explicit run directory")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. jiang.K=4")
    ap.add_argument("--pretrain-dqn", action="store_true")
    ap.add_argument("--pretrain-mode", choices=["setup", "xv"], default="setup",
                    help="setup: DQN accumulated per setup; xv: DQN carried across versions")
    ap.add_argument("--init-from", help="xv only: previous-version DQN to start from if none is stored yet")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if args.methods:
        cfg["methods"] = args.methods
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.rounds:
        cfg["training"]["rounds"] = args.rounds
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_path(cfg, k, v)
    log = lambda m: print(m, flush=True)

    if args.pretrain_dqn:
        pretrain_dqn(cfg, ROOT, cfg["seeds"], log=log, mode=args.pretrain_mode,
                     init_from=os.path.join(ROOT, args.init_from) if args.init_from else None)
        return
    out = args.out or os.path.join(ROOT, "runs", args.name or time.strftime("%Y%m%d-%H%M%S"))
    run(cfg, ROOT, out, args.workers, log)
    print(f"done -> {out}")


if __name__ == "__main__":
    sys.exit(main())
