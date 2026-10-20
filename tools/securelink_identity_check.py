"""Check that security.secure_link leaves training bit-identical (P17).

Short UAV-NIDD runs (seed 1, 20 rounds, voting from round 4; multi_krum + jiang_dqn) with the secure link off and on
with simulated interceptions (security.tamper_prob 0.3: about 30 % of packets rejected, NACKed and re-sent), serial and
with 4 parallel drones; compares rounds.jsonl (time_s ignored), summary.json (minutes ignored), global_model.pt, dqn.pt.
max_retries is raised to 10 here: at p = 0.3 a packet fails 4 attempts in a row with p^4 = 0.8 %, which over ~2000
packets per Jiang job would stop the run (the intended error); 0.3^11 * 2000 < 0.01.

  .venv/bin/python tools/securelink_identity_check.py [out_dir]     (default p17_scratch/identity)
"""
import json
import os
from collections import Counter
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from fldos.runner import run  # noqa: E402

METHODS = ["multi_krum", "jiang_dqn"]
TAMPER = {"secure_link": True, "tamper_prob": 0.3, "max_retries": 10}
VARIANTS = {"off": ({}, 2), "on_tamper": (TAMPER, 2), "on_tamper_par4": (TAMPER, 1)}


def config(variant):
    cfg = json.load(open(os.path.join(ROOT, "configs", "nidd_plain.json")))
    cfg.update(seeds=[1], methods=METHODS)
    cfg["training"]["rounds"] = 20
    cfg["jiang"]["vote_start_frac"] = 0.2
    sec, _ = VARIANTS[variant]
    if sec:
        cfg["security"] = dict(sec)
    if variant.endswith("_par4"):
        cfg["compute"]["parallel_drones"] = 4
    return cfg


def same_obj(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_obj(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_obj(x, y) for x, y in zip(a, b))
    return a == b or (a != a and b != b)  # NaN == NaN here


def compare(a, b):
    out = {}
    for m in METHODS:
        pa, pb = os.path.join(a, m, "seed_1"), os.path.join(b, m, "seed_1")
        ra = [{k: v for k, v in json.loads(l).items() if k != "time_s"} for l in open(os.path.join(pa, "rounds.jsonl"))]
        rb = [{k: v for k, v in json.loads(l).items() if k != "time_s"} for l in open(os.path.join(pb, "rounds.jsonl"))]
        sa = {k: v for k, v in json.load(open(os.path.join(pa, "summary.json"))).items() if k != "minutes"}
        sb = {k: v for k, v in json.load(open(os.path.join(pb, "summary.json"))).items() if k != "minutes"}
        r = {"rounds.jsonl": json.dumps(ra) == json.dumps(rb), "summary.json": json.dumps(sa) == json.dumps(sb),
             "global_model.pt": torch.equal(torch.load(os.path.join(pa, "global_model.pt")),
                                            torch.load(os.path.join(pb, "global_model.pt")))}
        if os.path.exists(os.path.join(pa, "dqn.pt")):
            r["dqn.pt"] = same_obj(torch.load(os.path.join(pa, "dqn.pt"), weights_only=False),
                                   torch.load(os.path.join(pb, "dqn.pt"), weights_only=False))
        out[m] = r
    return out


def crypto_stats(d):
    res = {}
    for m in METHODS:
        p = os.path.join(d, m, "seed_1", "securelink.jsonl")
        recs = [json.loads(l) for l in open(p)]
        rt = [json.loads(l)["time_s"] for l in open(os.path.join(d, m, "seed_1", "rounds.jsonl"))]
        res[m] = {"crypto_ms_per_round": round(sum(r["totals"]["crypto_ms"] for r in recs) / len(recs), 2),
                  "messages_per_round": round(sum(len(r["uplink"]) + len(r["downlink"]) + r["vote"]["messages"] + r["vote"].get("ballots", 0)
                                                  for r in recs) / len(recs), 1),
                  "MB_per_round": round(sum(r["totals"]["bytes"] for r in recs) / len(recs) / 1e6, 2),
                  "retransmits": sum(r.get("retransmits", 0) for r in recs),
                  "rejected_by": dict(sorted(Counter(e["error"] for r in recs for e in r.get("events", [])).items())),
                  "round_s": round(sum(rt) / len(rt), 3)}
    return res


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "p17_scratch", "identity")
    times = {}
    for v, (_, workers) in VARIANTS.items():
        t = time.time()
        run(config(v), ROOT, os.path.join(out, v), workers=workers, log=lambda m: None)
        times[v] = round(time.time() - t, 1)
        print(f"{v}: {times[v]} s", flush=True)
    report = {"wall_s": times,
              "on_tamper_vs_off": compare(os.path.join(out, "off"), os.path.join(out, "on_tamper")),
              "on_tamper_par4_vs_off": compare(os.path.join(out, "off"), os.path.join(out, "on_tamper_par4")),
              "crypto_on_tamper": crypto_stats(os.path.join(out, "on_tamper")),
              "crypto_on_tamper_par4": crypto_stats(os.path.join(out, "on_tamper_par4"))}
    print(json.dumps(report, indent=2))
    ok = all(all(r.values()) for k in ("on_tamper_vs_off", "on_tamper_par4_vs_off") for r in report[k].values())
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
