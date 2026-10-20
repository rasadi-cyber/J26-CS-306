"""P15b - ISOT input check (central, no FL): do more features help, and is the dataset's own CSV better / faster?

Arms (same rows of the same recordings, same split, same learners, same caps; only the INPUT differs):
  A29   our 29 size/timing features per packet (the v14m input)
  A29P  A29 + 13 per-packet protocol fields from the pcaps (tools/isot_proto_extract.py) + their window means
  C61   the ISOT CSV (new_feature_csv, 1 row = 10 packets), every column except ts
  CCL   C61 minus clock / identity columns, dropped by rule BEFORE any result (absolute times, ports, counters, TTL)
Splits:
  T     the pipeline's rolling split, fold = 103 % 5 (attacks: time split inside each recording; Normal: whole recordings)
  R     leak check: every attack type tested on WHOLE UNSEEN RECORDINGS (Normal as in T). A big T -> R drop means the
        input identifies the recording instead of the attack.
Traffic: pure recordings (all arms) and mixed traffic (A29, A29P only: the CSV rows were computed on pure captures
and cannot be re-mixed). CSV rows are matched to blocks of 10 packets, so packet arms are also scored per 10-packet
block (majority vote), the CSV's unit.
Learners: the FL model (MLP 256-128, dropout 0.1, inverse class weights, SGD lr 0.05, batch 256, best step on
validation) on MPS for every arm, and gradient-boosted trees (sklearn HGB, balanced weights) on CPU for every arm.
Model selection uses validation only; test is scored once.

  .venv/bin/python tools/feature_check.py     -> notes/logs15/feature_check.json (resumable per stage) + .md
"""
import glob, json, os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fldos.data import fold_segments, signed_log, window_features  # noqa: E402

SEED, FOLD = 103, 103 % 5
CLASSES = ["Normal", "UDP Flood", "TCP Flood", "HTTP Flood", "SlowHTTPTest", "Slowloris", "Other attack"]
C = len(CLASSES)
CAP = 200_000          # training rows per class (random, seed 103), every arm
STEPS, EVAL_EVERY = 30_000, 1_500
WINDOWS = [10, 100, 1000]
CSV_DIR = os.path.expanduser("~/Desktop/datasets/ISOT Drone Dataset/Dataset/new_feature_csv")
# Rule fixed before results: absolute clocks, addresses/ports, counters, host identity (TTL = CICIoT "Duration").
CSV_DROP_RULE = ["ts", "max_duration", "min_duration", "sum_duration", "average_duration", "Drone_port",
                 "Sequence number", "Duration"]
OUT = os.path.join(ROOT, "notes", "logs15", "feature_check.json")
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if os.environ.get("SMOKE"):  # quick end-to-end test of the script (tiny training), separate output
    STEPS, EVAL_EVERY, CAP = 300, 100, 3000
    OUT = OUT.replace(".json", "_smoke.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(res, open(OUT, "w"), indent=1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def roll_mean(x, grp, w):
    """Causal mean over the previous w rows of the same group (min_periods 1), rows in stream order."""
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    idx = np.arange(n)
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    gs = starts[np.searchsorted(starts, idx, "right") - 1]
    s = np.maximum(idx - w + 1, gs)
    return ((cs[idx + 1] - cs[s]) / (idx - s + 1)).astype(np.float32)


PROTO_WIN = ["is_tcp", "is_udp", "is_icmp", "syn", "ack", "fin", "rst", "psh", "payload_len", "web_port"]


def proto_windows(P, cols, grp):
    out, names = [], []
    for w in WINDOWS:
        for c in PROTO_WIN:
            out.append(roll_mean(P[:, cols.index(c)], grp, w)); names.append(f"pwin{w}.{c}")
    return np.column_stack(out), names


# ---------------------------------------------------------------- data
log("loading cleaned packet table")
import pyarrow.parquet as pq
tab = pq.read_table(os.path.join(ROOT, "data", "processed_isot", "uavnidd_clean.parquet"))
F29 = [c for c in tab.column_names if not c.startswith("_") and c != "Label"]
assert len(F29) == 29
order = tab.column("_order").to_numpy()
assert (np.diff(order) > 0).all()  # table is in capture order
cap_id, src = tab.column("_capture").to_numpy(), np.asarray(tab.column("_source").to_pylist(), dtype=str)
lab = np.asarray(tab.column("Label").to_pylist(), dtype=str)
y_pk = np.array([CLASSES.index(l) for l in np.unique(lab)])[np.unique(lab, return_inverse=True)[1]]
del lab
X29 = np.column_stack([tab.column(c).to_numpy().astype(np.float32) for c in F29])
del tab
df = pd.DataFrame({"_source": src, "_order": order, "_capture": cap_id})
man = json.load(open(os.path.join(ROOT, "data", "isot", "recordings.json")))["recordings"]
rstart = np.cumsum([0] + [r["packets_kept"] for r in man])
rec = np.searchsorted(rstart, order, "right") - 1
block_key = rec.astype(np.int64) * 10_000_000 + (order - rstart[rec]) // 10

seg_T = fold_segments(df, FOLD, 0.2, 0.1, ["Normal"], True)
del df


def unseen_split():
    """R: per attack type, whole recordings held out (closest to 20% of the type -> test, next closest to 10% ->
    validation if 3+ recordings, else the last 10% in time of the training recordings)."""
    seg = seg_T.copy()
    for s in np.unique(src):
        if s == "Normal":
            continue
        rows = np.flatnonzero(src == s)
        caps, cnt = np.unique(cap_id[rows], return_counts=True)
        real = caps[cnt >= 100]
        seg[rows] = 0
        if len(real) < 2:
            seg[rows] = seg_T[rows]  # single recording: cannot test unseen recordings (kept as T)
            continue
        sizes = {k: int((cap_id[rows] == k).sum()) for k in real}
        tot = sum(sizes.values())
        te = min(real, key=lambda k: abs(sizes[k] - 0.2 * tot))
        seg[rows[cap_id[rows] == te]] = 2
        rest = [k for k in real if k != te]
        if len(rest) >= 2:
            va = min(rest, key=lambda k: abs(sizes[k] - 0.1 * tot))
            seg[rows[cap_id[rows] == va]] = 1
        else:
            tr = rows[(seg[rows] == 0)]
            tr = tr[np.argsort(order[tr])]
            seg[tr[int(len(tr) * 0.9):]] = 1
    return seg


seg_R = unseen_split()
log("packets per split T", np.bincount(seg_T + 1, minlength=4).tolist(), "R", np.bincount(seg_R + 1, minlength=4).tolist())

log("protocol fields")
P_raw = np.load(os.path.join(ROOT, "data", "isot", "isot_proto.npy"), mmap_mode="r")
PCOLS = json.load(open(os.path.join(ROOT, "data", "isot", "isot_proto_cols.json")))
P = np.asarray(P_raw[order], dtype=np.float32)
PW, PWN = proto_windows(P, PCOLS, cap_id)
FP = F29 + PCOLS + PWN


def cap_rows(idx, y):
    rng = np.random.default_rng(SEED)
    out = [rng.permutation(idx[y[idx] == c])[:CAP] for c in range(C)]
    return np.sort(np.concatenate(out))


# Keep only the rows any arm uses (capped training rows of T and R, all validation and test rows): compact copies.
SPL = {}
for k, seg in (("T", seg_T), ("R", seg_R)):
    SPL[k] = (cap_rows(np.flatnonzero(seg == 0), y_pk), np.flatnonzero(seg == 1), np.flatnonzero(seg == 2))
need = np.unique(np.concatenate([a for t in SPL.values() for a in t]))
comp = {k: tuple(np.searchsorted(need, a) for a in t) for k, t in SPL.items()}
block_key_full = block_key
X29, P, y_pk, order, block_key = X29[need], P[need], y_pk[need], order[need], block_key[need]
XP = np.hstack([X29, P, PW[need]])
del PW
log("rows kept", len(need), "A29P features:", len(FP))

log("loading ISOT CSV")
bykey = {os.path.basename(r["recording"]): i for i, r in enumerate(man)}
# packet arrival times per recording (start time + gaps of the raw packet table), to place each CSV row by its ts
gaps_raw = pd.read_csv(os.path.join(ROOT, "data", "isot", "isot_packets.csv"), usecols=["frame.time_delta_displayed"],
                       engine="pyarrow")["frame.time_delta_displayed"].to_numpy(np.float64)
csv_fit = {}
frames = []
for p in sorted(glob.glob(os.path.join(CSV_DIR, "*", "*.csv"))):
    name = os.path.basename(p)[:-4]
    if name not in bykey:
        continue
    d = pd.read_csv(p)
    r = bykey[name]
    exp = int(np.ceil(man[r]["packets_kept"] / 10))
    assert abs(len(d) - exp) <= max(2, 0.01 * exp), (name, len(d), exp)
    t = man[r]["start_time"] + np.cumsum(gaps_raw[rstart[r]:rstart[r + 1]])
    i = np.clip(np.searchsorted(t, d["ts"].to_numpy(np.float64), "left"), 0, len(t) - 1)
    csv_fit[name] = {"rows": len(d), "blocks": exp, "ts_before_first_packet": int((d["ts"].to_numpy() < t[0] - 1).sum()),
                     "ts_after_last_packet": int((d["ts"].to_numpy() > t[-1] + 1).sum())}
    d["_key"] = r * 10_000_000 + i // 10
    lab = man[r]["label"]
    d["_y"] = CLASSES.index(lab) if lab in CLASSES else CLASSES.index("Other attack")
    frames.append(d)
csv = pd.concat(frames, ignore_index=True)
del frames
CCOLS = [c for c in csv.columns if c not in ("_key", "_y", "ts")]
CCL = [c for c in CCOLS if c not in CSV_DROP_RULE]
# CSV row -> split of its 10 packets (the first kept packet of the block); blocks with no kept packet are dropped
uk, first = np.unique(block_key_full, return_index=True)
pos = np.searchsorted(uk, csv["_key"].to_numpy())
ok = (pos < len(uk)) & (uk[np.minimum(pos, len(uk) - 1)] == csv["_key"].to_numpy())
csv = csv[ok].reset_index(drop=True)
pos = pos[ok]
cseg = {"T": seg_T[first[pos]], "R": seg_R[first[pos]]}
y_csv = csv["_y"].to_numpy()
XC = csv[CCOLS].to_numpy(np.float64)
XC = np.nan_to_num(XC, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
log("CSV rows", len(csv), "cols", len(CCOLS), "rule-clean cols", len(CCL))
del gaps_raw
res["csv_alignment"] = {"files": len(csv_fit), "rows": int(sum(v["rows"] for v in csv_fit.values())),
                        "blocks": int(sum(v["blocks"] for v in csv_fit.values())),
                        "rows_ts_outside_recording": int(sum(v["ts_before_first_packet"] + v["ts_after_last_packet"] for v in csv_fit.values())),
                        "rows_matched_to_kept_packets": int(ok.sum())}
res["csv_columns"] = CCOLS; res["csv_dropped_by_rule"] = [c for c in CSV_DROP_RULE if c in CCOLS + ["ts"]]
res["csv_clock_like_columns"] = [c for c in CCOLS if np.nanmedian(np.abs(XC[:, CCOLS.index(c)])) > 1e8]
save()


# ---------------------------------------------------------------- learners and metrics
def metrics(y, p):
    from sklearn.metrics import f1_score
    f1 = f1_score(y, p, labels=range(C), average=None, zero_division=0)
    nrm = y == 0
    att = ~nrm
    return {"macro_f1": float(f1.mean()), "f1": dict(zip(CLASSES, np.round(f1, 4).tolist())),
            "false_alarm": float((p[nrm] != 0).mean()) if nrm.any() else None,
            "missed_attack": float((p[att] == 0).mean()) if att.any() else None, "n": int(len(y))}


def block_vote(keys, y, p):
    uk_, inv = np.unique(keys, return_inverse=True)
    vy = np.zeros((len(uk_), C), np.int32); vp = np.zeros((len(uk_), C), np.int32)
    np.add.at(vy, (inv, y), 1); np.add.at(vp, (inv, p), 1)
    return vy.argmax(1), vp.argmax(1)


def prep(Xtr):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(signed_log(Xtr.astype(np.float64)))
    return lambda X: np.clip(sc.transform(signed_log(X.astype(np.float64))), -10, 10).astype(np.float32)


def class_w(y):
    cnt = np.bincount(y, minlength=C).astype(np.float64)
    w = np.zeros(C); pr = cnt > 0
    w[pr] = cnt[pr].sum() / (pr.sum() * cnt[pr]); w[pr] /= w[pr].mean()
    return torch.as_tensor(w, dtype=torch.float32, device=DEV)


def predict_mlp(net, X, bs=65536):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(net(torch.from_numpy(X[i:i + bs]).to(DEV)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def run_mlp(Xtr, ytr, Xva, yva, steps=None):
    steps = steps or STEPS
    torch.manual_seed(SEED)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
                        nn.Dropout(0.1), nn.Linear(128, C)).to(DEV)
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    w = class_w(ytr)
    Xt, yt = torch.from_numpy(Xtr).to(DEV), torch.from_numpy(ytr.astype(np.int64)).to(DEV)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    best, best_state, t0 = -1, None, time.time()
    for step in range(1, steps + 1):
        net.train()
        b = torch.randint(0, len(yt), (256,), generator=g).to(DEV)
        loss = F.cross_entropy(net(Xt[b]), yt[b], weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            m = metrics(yva, predict_mlp(net, Xva))["macro_f1"]
            if m > best:
                best, best_step = m, step
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, {"train_s": round(time.time() - t0, 1), "best_val_macro_f1": round(best, 4), "best_step": best_step}


def run_hgb(Xtr, ytr, es=True):
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    m = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63, class_weight="balanced", early_stopping=es,
                                       validation_fraction=0.1, random_state=SEED).fit(Xtr, ytr)
    return m, {"train_s": round(time.time() - t0, 1), "iters": int(m.n_iter_)}


def arm(name, X, y, splits, keys=None, mixed=None, steps=None, hgb_es=True):
    """Train both learners on one input/split and score test (row level, and per 10-packet block for packets)."""
    if name in res and res[name].get("done"):
        log("skip", name); return
    log("arm", name)
    tr, va, te = splits
    if mixed is not None:
        Xtr, ytr, Xva, yva, Xte, yte = mixed(tr, va, te)
        keys = None
    else:
        Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    tf = prep(Xtr)
    Xtr_s, Xva_s, Xte_s = tf(Xtr), tf(Xva), tf(Xte)
    out = {"n_features": int(X.shape[1]) if X is not None else int(Xtr.shape[1]), "n_train": int(len(ytr)),
           "train_per_class": np.bincount(ytr, minlength=C).tolist(), "n_test": int(len(yte))}
    net, info = run_mlp(Xtr_s, ytr, Xva_s, yva, steps)
    t0 = time.time(); p = predict_mlp(net, Xte_s)
    out["mlp"] = {**info, "test": metrics(yte, p), "predict_s": round(time.time() - t0, 1)}
    if keys is not None:
        out["mlp"]["test_block10"] = metrics(*block_vote(keys[te], yte, p))
    del net
    hgb, info = run_hgb(Xtr_s, ytr, hgb_es)
    t0 = time.time(); p = hgb.predict(Xte_s)
    out["hgb"] = {**info, "test": metrics(yte, p), "predict_s": round(time.time() - t0, 1),
                  "val_macro_f1": metrics(yva, hgb.predict(Xva_s))["macro_f1"]}
    if keys is not None:
        out["hgb"]["test_block10"] = metrics(*block_vote(keys[te], yte, p))
    out["done"] = True
    res[name] = out; save()
    log(name, "MLP", round(out["mlp"]["test"]["macro_f1"], 4), "HGB", round(out["hgb"]["test"]["macro_f1"], 4))


# ---------------------------------------------------------------- mixed traffic (packet arms only)
def build_mixed(rows, Pextra, rng):
    """fldos/traffic.build (same episodes, labels and window statistics), also carrying the per-packet protocol
    fields through the merge and adding their window means over the merged stream."""
    ln, gp, cl, od = X29[rows, 0].astype(float), X29[rows, 1].astype(float), y_pk[rows], order[rows]
    o = np.argsort(od, kind="stable")
    ln, gp, cl = ln[o], np.clip(gp[o], 0, None), cl[o]
    PE = Pextra[rows][o] if Pextra is not None else None
    is_n = cl == 0
    n_len, n_gap = ln[is_n], gp[is_n]
    n_P = PE[is_n] if PE is not None else None
    ptr, X, Y, EP, PP, ep = 0, [], [], [], [], 0
    for c in [c for c in np.unique(cl) if c != 0]:
        for s, e in _bursts(int((cl == c).sum()), rng, 200, 2000):
            sel = np.flatnonzero(cl == c)[s:e]
            b_len, b_gap = ln[sel], gp[sel]
            m = max(20, int(len(b_len) * rng.uniform(0.5, 2.0)))
            take = (ptr + np.arange(m)) % len(n_len); ptr = (ptr + m) % len(n_len)
            t_bg = np.cumsum(n_gap[take])
            start = rng.uniform(0.1, 0.6) * max(t_bg[-1] if m else 0.0, 1e-6)
            t = np.concatenate([t_bg, start + np.cumsum(b_gap)])
            L = np.concatenate([n_len[take], b_len]); att = np.concatenate([np.zeros(m, bool), np.ones(len(sel), bool)])
            srt = np.argsort(t, kind="stable"); t, L, att = t[srt], L[srt], att[srt]
            if PE is not None:
                PP.append(np.vstack([n_P[take], PE[sel]])[srt])
            g = np.diff(t, prepend=t[0])
            cs = np.cumsum(att); win = np.minimum(np.arange(1, len(att) + 1), 10)
            share = (cs - np.concatenate([np.zeros(10), cs[:-10]])[:len(cs)]) / win
            X.append(np.column_stack([L, g])); Y.append(np.where(share >= 0.5, c, 0)); EP.append(np.full(len(t), ep)); ep += 1
    X, Y, EP = np.vstack(X), np.concatenate(Y), np.concatenate(EP)
    W, _ = window_features(X, EP, ["frame.len", "frame.time_delta_displayed"], WINDOWS, full=True)
    out = np.hstack([X, W]).astype(np.float32)
    if PE is not None:
        PPm = np.vstack(PP)
        PWm, _ = proto_windows(PPm, PCOLS, EP)
        out = np.hstack([out, PPm, PWm])
    return out, Y.astype(np.int64)


def _bursts(n, rng, lo, hi):
    cuts, i = [], 0
    while i < n:
        b = int(rng.integers(lo, hi + 1)); cuts.append((i, min(n, i + b))); i += b
    return cuts


def mixed_fn(with_proto):
    def f(tr, va, te):
        out = []
        for k, rows in enumerate((tr, va, te)):
            Xm, ym = build_mixed(rows, P if with_proto else None, np.random.default_rng(SEED + 12345 + k))
            out += [Xm, ym]
        return out
    return f


# ---------------------------------------------------------------- run
def csv_splits(seg):
    return cap_rows(np.flatnonzero(seg == 0), y_csv), np.flatnonzero(seg == 1), np.flatnonzero(seg == 2)


for split in ("T", "R"):
    arm(f"A29_pure_{split}", X29, y_pk, comp[split], keys=block_key)
    arm(f"A29P_pure_{split}", XP, y_pk, comp[split], keys=block_key)
    arm(f"C61_pure_{split}", XC, y_csv, csv_splits(cseg[split]))
    arm(f"CCL_pure_{split}", XC[:, [CCOLS.index(c) for c in CCL]], y_csv, csv_splits(cseg[split]))
arm("A29_mixed_T", None, y_pk, comp["T"], mixed=mixed_fn(False))
arm("A29P_mixed_T", None, y_pk, comp["T"], mixed=mixed_fn(True))
# Is the MLP's mixed-traffic gap (29 inputs) under-training, and is the trees' drop with protocol fields an
# early-stopping artefact? Same data, 4x the MLP steps, trees without early stopping (300 iterations).
arm("A29_mixed_T_long", None, y_pk, comp["T"], mixed=mixed_fn(False), steps=4 * STEPS, hgb_es=False)
arm("A29P_mixed_T_long", None, y_pk, comp["T"], mixed=mixed_fn(True), steps=4 * STEPS, hgb_es=False)

# Single-column leak screen (split R = unseen recordings): a column that alone separates the classes on unseen
# recordings carries behaviour; a column that is strong on T but collapses on R identifies the recording.
if not res.get("single_column", {}).get("done") and not os.environ.get("SMOKE"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    sc = {}
    rng = np.random.default_rng(SEED)
    def sub(idx, y, k):
        return np.sort(np.concatenate([rng.permutation(idx[y[idx] == c])[:k] for c in range(C)]))
    idx = {"csv": {k: (np.flatnonzero(cseg[k] == 0), np.flatnonzero(cseg[k] == 2)) for k in ("T", "R")},
           "packet": {k: (comp[k][0], comp[k][2]) for k in ("T", "R")}}
    for tag, X, y, names in (("csv", XC, y_csv, CCOLS), ("packet", XP, y_pk, FP)):
        for j, nm in enumerate(names):
            row = {}
            for split in ("T", "R"):
                tr, te = sub(idx[tag][split][0], y, 20_000), sub(idx[tag][split][1], y, 20_000)
                m = HistGradientBoostingClassifier(max_iter=60, class_weight="balanced", random_state=SEED)
                m.fit(X[tr, j:j + 1], y[tr])
                row[split] = round(metrics(y[te], m.predict(X[te, j:j + 1]))["macro_f1"], 4)
            sc[f"{tag}:{nm}"] = row
        log("single-column done", tag)
    res["single_column"] = {"done": True, "macro_f1_balanced_subsample": sc}; save()
log("ALL DONE")
