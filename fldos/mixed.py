"""Mixed-traffic test: a separate, more realistic evaluation of already-trained global models.

The standard test computes window features over one recording at a time, so every window holds a single class.
A drone's monitor would see normal traffic and an attack interleaved. Here each episode is built from the test
split only:
  background  all Normal test packets, on their own inter-arrival timeline
  burst       a run of one attack type's test packets, inserted at a random time in that timeline
The two streams are merged by arrival time, and the inter-arrival times and window statistics are recomputed over
the merged stream, so a window can contain both normal and attack packets. Models are not retrained.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from .data import ORDER_COL, SEGMENT_COL, LABEL_COL, make_partition, signed_log, window_features
from .model import MLP, set_vector

BURST = 2000          # attack packets per episode
PLACEMENTS = 3        # random burst positions per attack type


def _stream(df, label):
    rows = df[(df[LABEL_COL] == label) & (df[SEGMENT_COL] == 2)].sort_values(ORDER_COL)
    return rows["frame.len"].to_numpy(float), np.clip(rows["frame.time_delta_displayed"].to_numpy(float), 0, None)


def build_episodes(clean, classes, normal, window_sizes, seed=0):
    """Return (X raw features, y labels, episode id) over all episodes."""
    rng = np.random.default_rng(seed)
    n_len, n_dt = _stream(clean, normal)
    n_t = np.cumsum(n_dt)
    Xs, ys, eps = [], [], []
    ep = 0
    for c in [c for c in classes if c != normal]:
        a_len, a_dt = _stream(clean, c)
        if len(a_len) == 0:
            continue
        for _ in range(PLACEMENTS):
            k = int(rng.integers(0, max(1, len(a_len) - BURST)))
            b_len, b_dt = a_len[k:k + BURST], a_dt[k:k + BURST]
            start = rng.uniform(0.2, 0.6) * n_t[-1]  # burst begins somewhere in the first half of the background
            b_t = start + np.cumsum(b_dt)
            t = np.concatenate([n_t, b_t])
            ln = np.concatenate([n_len, b_len])
            lab = np.array([normal] * len(n_t) + [c] * len(b_t))
            o = np.argsort(t, kind="stable")
            t, ln, lab = t[o], ln[o], lab[o]
            dt = np.diff(t, prepend=t[0])
            X = np.column_stack([ln, dt])
            W, _ = window_features(X, np.zeros(len(t), dtype=int), ["frame.len", "frame.time_delta_displayed"],
                                   window_sizes, full=True)
            Xs.append(np.hstack([X, W])); ys.append(lab); eps.append(np.full(len(t), ep)); ep += 1
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(eps)


def evaluate(cfg, root, run_dir, log=print):
    dc, tc = cfg["data"], cfg["training"]
    clean = pd.read_csv(os.path.join(root, dc["processed_dir"], "uavnidd_clean.csv"))
    classes = list(dc["classes"]) + ([dc["other_attacks_as"]] if dc.get("other_attacks_as") else [])
    X_raw, y_lab, _ = build_episodes(clean, classes, dc["normal_class"], dc["window_sizes"])
    y = np.array([classes.index(l) for l in y_lab])
    normal_t, rare_t, C = classes.index(dc["normal_class"]), classes.index(dc["rare_class"]), len(classes)
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    out = {"episodes": {"burst_packets": BURST, "placements_per_attack": PLACEMENTS, "rows": int(len(y)),
                        "rows_per_class": {c: int((y == i).sum()) for i, c in enumerate(classes)}},
           "class_names": classes, "methods": {}}
    for seed in meta["seeds"]:
        data = make_partition(cfg, seed, root, lambda m: None, save_files=False)
        X = np.clip(data.scaler.transform(signed_log(X_raw)), -10, 10).astype(np.float32)
        Xt = torch.as_tensor(X)
        model = MLP(X.shape[1], C, tc["hidden"], tc["dropout"])
        for m in meta["methods"]:
            p = os.path.join(run_dir, m, f"seed_{seed}", "global_model.pt")
            if not os.path.exists(p):
                continue
            set_vector(model, torch.load(p, weights_only=False))
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(Xt[i:i + 65536]).argmax(1) for i in range(0, len(Xt), 65536)]).numpy()
            cm = np.zeros((C, C), dtype=np.int64)
            np.add.at(cm, (y, pred), 1)
            e = out["methods"].setdefault(m, {"confusion": np.zeros((C, C), dtype=np.int64).tolist(), "per_seed": []})
            e["confusion"] = (np.array(e["confusion"]) + cm).tolist()
            tp = np.diag(cm).astype(float)
            prec = np.divide(tp, cm.sum(0), out=np.zeros(C), where=cm.sum(0) > 0)
            rec = np.divide(tp, cm.sum(1), out=np.zeros(C), where=cm.sum(1) > 0)
            f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(C), where=(prec + rec) > 0)
            att = y != normal_t
            e["per_seed"].append({"seed": seed, "macro_f1": float(f1.mean()), "accuracy": float(tp.sum() / len(y)),
                                  "rare_f1": float(f1[rare_t]),
                                  "missed_attack_rate": float((pred[att] == normal_t).mean()),
                                  "false_alarm_rate": float((pred[~att] != normal_t).mean()),
                                  "per_class_f1": {classes[k]: float(f1[k]) for k in range(C)}})
        log(f"mixed-traffic test: seed {seed} done")
    for m, e in out["methods"].items():
        for k in ("macro_f1", "accuracy", "rare_f1", "missed_attack_rate", "false_alarm_rate"):
            v = [s[k] for s in e["per_seed"]]
            e[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    with open(os.path.join(run_dir, "mixed_traffic.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out
