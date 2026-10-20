"""Evaluation extras for finished runs (no training): the exact data partition per seed and an out-of-time test.

For each run and seed it rebuilds that seed's split with the same code and settings as training
(fldos.data._make_partition with `extras`, which stops right after the split) and writes, next to the run:
  partition_report.json  {seed: exact row counts per class for the drones' pool, each drone's training (80%) and
                          voting (20%) rows, the public exam, validation, test and the out-of-time slice}
  out_of_time.json       {seed: {method: {"test": {...}, "future": {...}}}}: the final global model scored on the
                          test window and on the later slice (traffic after the test window, never trained or tested
                          on), both as per-capture windows so the two are comparable; per attack type: rows and recall.
The split is checked against the drone counts the run recorded in meta.json (it must match exactly).

  .venv/bin/python tools/eval_extras.py --runs v14m-NIDD-CPU-FINAL NIDD-plain-FINAL
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fldos.data import _make_partition  # noqa: E402
from fldos.model import MLP, set_vector  # noqa: E402

_CACHE = {}


def extras(cfg, seed):
    key = hashlib.md5(json.dumps({"data": cfg["data"], "clients": cfg["clients"], "seed": seed},
                                 sort_keys=True).encode()).hexdigest()
    if key not in _CACHE:
        _CACHE.clear()  # one partition in memory at a time (ISOT is large)
        d = {}
        _make_partition(cfg, seed, ROOT, log=lambda m: None, save_files=False, extras=d)
        _CACHE[key] = d
    return _CACHE[key]


def predict(cfg, vec, X, C):
    model = MLP(X.shape[1], C, cfg["training"]["hidden"], cfg["training"]["dropout"])
    set_vector(model, vec.float())
    model.eval()
    if (cfg.get("compute") or {}).get("feature_dtype") == "float16":  # as the run stored and scored its features
        X = X.astype(np.float16)
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(X[i:i + 65536]).float()).argmax(1).numpy()
                               for i in range(0, len(X), 65536)]) if len(X) else np.array([], dtype=int)


def scores(y, p, classes, normal_t):
    out = {}
    for t, c in enumerate(classes):
        m = y == t
        if t == normal_t or not m.any():
            continue
        out[c] = {"rows": int(m.sum()), "recall": float((p[m] == t).mean()),
                  "detected_as_attack": float((p[m] != normal_t).mean())}
    att = y != normal_t
    out["_all_attacks"] = {"rows": int(att.sum()),
                           "detected_as_attack": float((p[att] != normal_t).mean()) if att.any() else None}
    return out


def run_one(run):
    rd = os.path.join(ROOT, "runs", run)
    meta = json.load(open(os.path.join(rd, "meta.json")))
    cfg = meta["config"]
    rep, oot = {}, {}
    for seed in meta["seeds"]:
        d = extras(cfg, seed)
        r = d["report"]
        # check: the rebuilt split is the run's split (drone class counts recorded at training time)
        recorded = {c["cid"]: c["counts"] for c in meta["partitions"][str(seed)]["clients"]}
        for dr in r["drones"]:
            tot = {k: dr["train"].get(k, 0) + dr["vote"].get(k, 0) for k in r["classes"]}
            assert tot == {k: recorded[dr["cid"]].get(k, 0) for k in r["classes"]}, (run, seed, dr["cid"])
        rep[str(seed)] = r
        classes, nt = d["classes"], d["normal_t"]
        (Xt, yt), (Xf, yf) = d["test_pure"], d["future"]
        oot[str(seed)] = {}
        for m in meta["methods"]:
            path = os.path.join(rd, m, f"seed_{seed}", "global_model.pt")
            if not os.path.exists(path):
                continue
            vec = torch.load(path)
            oot[str(seed)][m] = {"test": scores(yt, predict(cfg, vec, Xt, len(classes)), classes, nt),
                                 "future": scores(yf, predict(cfg, vec, Xf, len(classes)), classes, nt)}
        print(f"{run} seed {seed}: out-of-time rows {len(yf):,} (test window rows {len(yt):,})", flush=True)
    for name, obj in (("partition_report.json", rep), ("out_of_time.json", oot)):
        tmp = os.path.join(rd, name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, os.path.join(rd, name))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    for r in ap.parse_args().runs:
        run_one(r)
