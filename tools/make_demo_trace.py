"""Build data/demo/demo_trace.json for the Secure Link demo (notes/SPEC_P17.md, "Demo trace") from a REAL short run:
UAV-NIDD plain config, seed 1, Jiang + DQN, 40 rounds, voting from round 8 (vote_start_frac 0.2), secure link on,
10 parallel drones, simulated interceptions on a few packets (security.tamper_prob = TAMPER_PROB; each rejected packet
is NACKed and re-sent, results unchanged). Validation is scored every round (eval_every 1) so the demo can show it each
round; validation never feeds training, so the trained models are the same as with the config's cadence.
The trace's `tamper_demo` is a REAL interception from the run (original packet, flipped byte, rejection, NACK, re-sent
packet, accepted bit-identical); only a run without any uplink interception falls back to a throwaway enrollment.

  .venv/bin/python tools/make_demo_trace.py                  # new run in runs/P17-demo-trace2, then the trace
  .venv/bin/python tools/make_demo_trace.py --run-dir DIR    # trace from an existing secure-link run
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from fldos import securelink  # noqa: E402
from fldos.runner import run  # noqa: E402

CONFIG = "configs/nidd_plain.json"
SEED, METHOD, ROUNDS = 1, "jiang_dqn", 40
TAMPER_PROB = 0.005  # ~0.5 % of the ~2300 data packets of the run: a handful of interceptions
RUN_NAME = "P17-demo-trace2"
# 10 generic site labels on a small local grid (2 rows x 5 columns, ~1.1 km apart); x, y in [0, 1] for drawing
SITES = ["Site A - North ridge", "Site B - Quarry road", "Site C - River bend", "Site D - Solar farm",
         "Site E - Harbour gate", "Site F - Rail yard", "Site G - West pass", "Site H - Forest edge",
         "Site I - Water works", "Site J - South depot"]
LAT0, LON0, STEP = 6.900, 79.900, 0.010
SERVER = {"name": "Aggregation server - Operations centre", "lat": round(LAT0 - 0.5 * STEP, 5),
          "lon": round(LON0 + 2.0 * STEP, 5), "x": 0.5, "y": 0.5}
VAL_KEYS = ("macro_f1", "rare_recall", "missed_attack_rate", "accuracy", "false_alarm_rate")


def config():
    cfg = json.load(open(os.path.join(ROOT, CONFIG)))
    cfg.update(seeds=[SEED], methods=[METHOD])
    cfg["training"]["rounds"] = ROUNDS
    cfg["training"]["eval_every"] = 1
    cfg["jiang"]["vote_start_frac"] = 0.2
    cfg["compute"]["parallel_drones"] = 10
    cfg["security"] = {"secure_link": True, "bandwidth_mbps": 10.0, "latency_ms": 20.0, "tamper_prob": TAMPER_PROB,
                       "max_retries": 3}
    return cfg


def site(cid):
    row, col = divmod(cid, 5)
    return {"name": SITES[cid], "lat": round(LAT0 - row * STEP, 5), "lon": round(LON0 + col * STEP, 5),
            "x": round(0.1 + 0.2 * col, 3), "y": round(0.2 + 0.6 * row, 3)}


def r6(xs):
    return [round(x, 6) for x in xs]


def _event(e):
    """Compact per-round interception record (the full hex sequence is kept for the demo's one event only)."""
    return {k: e.get(k) for k in ("cid", "dir", "attempt", "error", "byte", "bytes", "nack_bytes", "resend_bytes",
                                  "recovered")}


def pick_event(links, rounds_with_vote):
    """The interception the demo shows: an uplink packet rejected on its first attempt and accepted on the re-send,
    preferably in a voting round (the demo's last part), else any uplink interception."""
    evs = [(r, e) for r, sl in sorted(links.items()) for e in sl.get("events", [])
           if e["dir"] == "uplink" and e["attempt"] == 1 and e.get("recovered")]
    pool = [x for x in evs if x[0] in rounds_with_vote] or evs
    return pool[-1] if pool else None


def build(run_dir, method=METHOD, seed=None):
    """Demo trace from any secure-link run folder (its files only): method/seed job, partition, link log."""
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    seed = seed if seed is not None else (meta.get("seeds") or [SEED])[0]
    job = os.path.join(run_dir, method, f"seed_{seed}")
    cfg = meta["config"]
    rounds = [json.loads(l) for l in open(os.path.join(job, "rounds.jsonl"))]
    links = {r["round"]: r for r in map(json.loads, open(os.path.join(job, "securelink.jsonl")))}
    crypto = json.load(open(os.path.join(job, "securelink_meta.json")))
    part = meta["partitions"][str(seed)]
    sites = [{"cid": c["cid"], **site(c["cid"]), "role": c["role"], "n_train": c["n_train"],
              "n_vote_set": c["n_vote_set"], "counts": c["counts"]} for c in part["clients"]]
    out_rounds = []
    for rec in rounds:
        sl = links[rec["round"]]
        up = sl["uplink"]
        for u in up:
            u["plain_sample"], u["decrypted_sample"] = r6(u["plain_sample"]), r6(u["decrypted_sample"])
        rows = []
        for d in rec["clients"]:
            row = {"cid": d["cid"], "role": d["role"], "status": d["status"], "weight": round(d["weight"], 6),
                   "score": round(d["score"], 4), "filtered_out": d.get("filtered_out", False)}
            if "votes_yes" in d:
                row.update(yes=d["votes_yes"], votes_total=d["votes_total"], readmitted=d["readmitted"])
            if d.get("dqn_weight") is not None:
                row.update(dqn_weight=d["dqn_weight"], reward=d.get("reward"))
            rows.append(row)
        val = rec["val"] or {}
        out_rounds.append({
            "round": rec["round"],
            "local": [{"cid": u["cid"], "n_steps": cfg["training"]["local_steps"], "local_train_ms": u["local_train_ms"],
                       "update_norm": u["update_norm"], "plain_sample": u["plain_sample"]} for u in up],
            "uplink": up,
            "server": {"rows": rows,
                       "selected": [d["cid"] for d in rec["clients"] if not d.get("filtered_out")],
                       "dropped": [d["cid"] for d in rec["clients"] if d.get("filtered_out")],
                       "readmitted": [d["cid"] for d in rec["clients"] if d.get("readmitted")],
                       "dqn_weights": {str(d["cid"]): d["dqn_weight"] for d in rec["clients"]
                                       if d.get("dqn_weight") is not None},
                       "voting_active": rec["info"].get("voting_active"), "epsilon": rec["info"].get("epsilon"),
                       "global_sample": r6(sl["global_sample"]),
                       "val": {k: round(val[k], 4) for k in VAL_KEYS if k in val}},
            "downlink": sl["downlink"],
            "vote": sl["vote"], "totals": sl["totals"],
            "retransmits": sl.get("retransmits", 0), "events": [_event(e) for e in sl.get("events", [])]})
    # Interception -> NACK -> re-send: a real event of this run (the uplink record holds the accepted re-send).
    voting = {r["round"] for r in rounds if r["info"].get("voting_active")}
    hit = pick_event(links, voting)
    if hit:
        r, ev = hit
        acc = next(u for u in links[r]["uplink"] if u["cid"] == ev["cid"])
        tamper = {**securelink.event_story(dict(ev, round=r), acc), "source": "run"}
    else:  # no uplink interception in this run: a separate throwaway enrollment, plaintext = the final model
        g = torch.load(os.path.join(job, "global_model.pt"))
        honest = [s["cid"] for s in sites if s["role"] == "general"]
        tamper = securelink.tamper_demo(g, cid=honest[0], rnd=len(rounds) - 1)
    crypto.pop("totals", None)
    cfg_file = os.path.join(run_dir, "config.json")
    dataset = "ISOT Drone" if "isot" in str(cfg.get("data", {}).get("csv_path", "")).lower() else "UAV-NIDD"
    trace = {"meta": {"dataset": dataset, "config_file": os.path.relpath(cfg_file, ROOT) if os.path.exists(cfg_file) else CONFIG,
                      "seed": seed, "method": method,
                      "rounds": len(rounds), "vote_start": meta["vote_start_round"], "class_names": meta["class_names"],
                      "rare_class": meta["rare_class"], "local_steps": cfg["training"]["local_steps"],
                      "eval_every": cfg["training"]["eval_every"], "run_dir": os.path.relpath(run_dir, ROOT),
                      "K": cfg["jiang"]["K"], "vote_start_frac": cfg["jiang"]["vote_start_frac"],
                      "num_clients": cfg["clients"]["num_clients"], "num_poisoned": cfg["clients"]["num_poisoned"],
                      "tamper_prob": (cfg.get("security") or {}).get("tamper_prob", 0.0),
                      "max_retries": (cfg.get("security") or {}).get("max_retries", 3),
                      "server": SERVER, "sites": sites, "crypto": crypto},
             "rounds": out_rounds, "tamper_demo": tamper}
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help=f"existing secure-link run (default: make runs/{RUN_NAME})")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "demo", "demo_trace.json"))
    a = ap.parse_args()
    run_dir = a.run_dir or os.path.join(ROOT, "runs", RUN_NAME)
    if not a.run_dir:
        run(config(), ROOT, run_dir, workers=1, log=lambda m: print(m, flush=True))
    trace = build(run_dir)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(trace, f, separators=(",", ":"))
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.2f} MB, {len(trace['rounds'])} rounds")


if __name__ == "__main__":
    main()
