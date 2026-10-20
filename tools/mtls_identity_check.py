"""Check that security.transport = "mtls" leaves training bit-identical (P18).

Config demo_runs/encrypted.json (UAV-NIDD, FedAvg + Jiang + DQN, seed 1) shortened to 20 rounds (voting from round 4):
  inproc       no link, drones trained in the job process
  mtls         real mTLS (10 drone processes, gRPC over TLS 1.3) + message-level secure link, no interceptions
  mtls_tamper  as mtls with security.tamper_prob 0.3 (max_retries 10, see tools/securelink_identity_check.py)
Compares rounds.jsonl (time_s ignored), summary.json (minutes ignored), global_model.pt, dqn.pt against inproc.
One job at a time (each mTLS job runs 10 drone processes).

  .venv/bin/python tools/mtls_identity_check.py [out_dir]     (default runs/P18-mtls-identity; delete it afterwards)
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fldos.runner import run  # noqa: E402
from securelink_identity_check import compare  # noqa: E402  (same comparison as P17; METHODS patched below)
import securelink_identity_check as sic  # noqa: E402

METHODS = ["fedavg", "jiang_dqn"]
sic.METHODS = METHODS
VARIANTS = {"inproc": None,
            "mtls": {"transport": "mtls", "tamper_prob": 0.0},
            "mtls_tamper": {"transport": "mtls", "tamper_prob": 0.3, "max_retries": 10}}


def config(variant):
    cfg = json.load(open(os.path.join(ROOT, "demo_runs", "encrypted.json")))
    cfg.update(seeds=[1], methods=METHODS)
    cfg["training"]["rounds"] = 20
    cfg["jiang"]["vote_start_frac"] = 0.2
    if VARIANTS[variant] is None:
        cfg.pop("security", None)
    else:
        cfg["security"].update(VARIANTS[variant])
    return cfg


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "P18-mtls-identity")
    times = {}
    for v in VARIANTS:
        t = time.time()
        run(config(v), ROOT, os.path.join(out, v), workers=1, log=lambda m: None)
        times[v] = round(time.time() - t, 1)
        print(f"{v}: {times[v]} s", flush=True)
    report = {"wall_s": times}
    for v in ("mtls", "mtls_tamper"):
        report[f"{v}_vs_inproc"] = compare(os.path.join(out, "inproc"), os.path.join(out, v))
        d = os.path.join(out, v, "jiang_dqn", "seed_1")
        meta = json.load(open(os.path.join(d, "mtls_meta.json")))
        lines = [json.loads(l) for l in open(os.path.join(d, "mtls.jsonl"))]
        sl = [json.loads(l) for l in open(os.path.join(d, "securelink.jsonl"))]
        report[f"{v}_jiang_link"] = {
            "drones": sorted({(x["status"], x["tls_version"], x["cipher"]) for x in meta["drones"]}),
            "handshake_ms": [x["handshake_ms"] for x in meta["drones"]],
            "MB_per_round": round(sum(d["bytes_up"] + d["bytes_down"] for x in lines for d in x["drones"])
                                  / len(lines) / 1e6, 2),
            "rounds_logged": len(lines), "retransmits": sum(r.get("retransmits", 0) for r in sl)}
    print(json.dumps(report, indent=2))
    ok = all(all(r.values()) for k in ("mtls_vs_inproc", "mtls_tamper_vs_inproc") for r in report[k].values())
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
