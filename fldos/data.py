"""UAV-NIDD cleaning, splitting and federated partitioning (Jiang et al. setup adapted to DoS detection).

Pipeline
  1. load_raw       parse the CSV once (hex -> int, IP/MAC strings -> 0), cached as .npz
  2. build_clean    keep the configured classes, drop identifier columns, drop exact duplicates
                    -> data/processed/uavnidd_clean.csv + clean_report.json
  3. make_partition per seed: test / validation / public exam / client pool, then 10 drones
                    -> data/processed/seed_<s>/ (parquet files + manifest.json)

Drone roles (paper Sec. 5.1): general drones share the common classes through a Dirichlet split;
the rare drone additionally owns every training row of the rare class; the poisoned drone holds an
ordinary common share but flips its labels. Every drone's local test set is topped up with the
public exam (a small class-balanced third-party sample), so all voters can judge every class.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_COL = "Label"
ORDER_COL = "_order"  # raw row index = capture order; used for the time-based split, never as a feature
SEGMENT_COL = "_segment"  # 0 drones / 1 validation / 2 test, fixed on the raw capture order (temporal split)
CAPTURE_COL = "_capture"  # capture file id (packet numbers continue +1 within a capture); found without labels
SOURCE_COL = "_source"  # original UAV-NIDD label before merging into "Other attack"; split bookkeeping only
RAW_CACHE_VERSION = 2


def time_segments(labels, test_frac, val_frac, frame_number=None):
    """Per original attack type, in capture order: 0 = earliest (drones), 1 = next (validation), 2 = latest (test).
    A type recorded as 3+ separate captures (frame numbers restart) is split by capture instead: last capture ->
    test, the one before -> validation, the rest -> drones, so validation and test are captures never trained on."""
    seg = np.zeros(len(labels), dtype=np.int8)
    for c in np.unique(labels):
        rows = np.flatnonzero(labels == c)
        if frame_number is not None:
            cap = np.concatenate([[0], np.cumsum(np.diff(frame_number[rows]) < 0)])
            if cap.max() >= 2:
                seg[rows[cap == cap.max() - 1]] = 1
                seg[rows[cap == cap.max()]] = 2
                continue
        n_te, n_va = int(round(len(rows) * test_frac)), int(round(len(rows) * val_frac))
        seg[rows[len(rows) - n_te - n_va:len(rows) - n_te]] = 1
        seg[rows[len(rows) - n_te:]] = 2
    return seg


def fold_segments(df, fold, test_frac, val_frac, rotate_only=None, attacks_per_capture=False):
    """Rolling-origin split (fold 0-4): per original attack type, in capture order, the test block starts at
    60/65/70/75/80% of the timeline, validation is the block just before it, the drones get everything earlier.
    A type recorded as 3+ separate captures (Normal) rotates whole captures instead: capture `fold` -> test,
    the next one -> validation, the rest -> drones. Every test packet is later than (or from another capture than)
    every training packet, and each fold tests on different traffic."""
    seg = np.full(len(df), -1, dtype=np.int8)
    src = df[SOURCE_COL].to_numpy(dtype=str)
    order = df[ORDER_COL].to_numpy()
    cap = df[CAPTURE_COL].to_numpy()
    for c in np.unique(src):
        rows = np.flatnonzero(src == c)
        rows = rows[np.argsort(order[rows], kind="stable")]
        # real captures (>= 100 rows) in capture order; stray rows left by duplicate copies stay with the drones
        caps = [k for k in dict.fromkeys(cap[rows]) if (cap[rows] == k).sum() >= 100]
        # rotate_only (e.g. ["Normal"]): only these types rotate whole captures; other types are split in time.
        # attacks_per_capture: the time split is applied inside each capture, so the test holds the later part of
        # every attack session (what the split means for UAV-NIDD, where each attack type is a single capture).
        rotate = len(caps) >= 3 and (rotate_only is None or c in rotate_only)
        if not rotate and attacks_per_capture and len(caps) >= 2:
            for k in dict.fromkeys(cap[rows]):
                r = rows[cap[rows] == k]
                n = len(r)
                t0 = int(round(n * (0.60 + 0.05 * fold)))
                te_end = min(n, t0 + int(round(n * test_frac)))
                va0 = max(0, t0 - int(round(n * val_frac)))
                seg[r[:va0]] = 0
                seg[r[va0:t0]] = 1
                seg[r[t0:te_end]] = 2
            continue
        if rotate:
            seg[rows] = 0
            seg[rows[cap[rows] == caps[fold % len(caps)]]] = 2
            seg[rows[cap[rows] == caps[(fold + 1) % len(caps)]]] = 1
            continue
        n = len(rows)
        t0 = int(round(n * (0.60 + 0.05 * fold)))
        te_end = min(n, t0 + int(round(n * test_frac)))
        va0 = max(0, t0 - int(round(n * val_frac)))
        seg[rows[:va0]] = 0
        seg[rows[va0:t0]] = 1
        seg[rows[t0:te_end]] = 2
    return seg  # -1 = later than this fold's test block: unused


def window_features(X, labels, feats, sizes, segment=None, full=False):
    """Causal sliding-window statistics over the previous W packets of the same capture stream (Kitsune-style,
    Mirsky et al. NDSS 2018). Basic: mean/std of frame length and inter-arrival time, packet rate. Full adds
    min/max frame length, max gap and byte rate. A stream is a contiguous run of rows with one label in the raw
    file (rows are in capture order). Computed before de-duplication, because repeated packets are part of the
    real traffic rate."""
    ln = X[:, feats.index("frame.len")]
    dt = np.clip(X[:, feats.index("frame.time_delta_displayed")], 0, None)
    stream = np.concatenate([[0], np.cumsum(labels[1:] != labels[:-1])])
    if segment is not None:  # restart every window at a split boundary: no test window holds a training packet
        stream = stream * 3 + segment
    df = pd.DataFrame({"s": stream, "len": ln, "dt": dt})
    g = df.groupby("s")
    cols, names = [], []
    for w in sizes:
        r = lambda c, f: getattr(g[c].rolling(w, min_periods=1), f)().reset_index(level=0, drop=True).sort_index()
        len_m, len_s, dt_m, dt_s = r("len", "mean"), r("len", "std").fillna(0), r("dt", "mean"), r("dt", "std").fillna(0)
        cols += [len_m, len_s, dt_m, dt_s, 1.0 / (dt_m + 1e-6)]
        names += [f"win{w}.len_mean", f"win{w}.len_std", f"win{w}.dt_mean", f"win{w}.dt_std", f"win{w}.rate"]
        if full:
            cols += [r("len", "min"), r("len", "max"), r("dt", "max"), len_m / (dt_m + 1e-6)]
            names += [f"win{w}.len_min", f"win{w}.len_max", f"win{w}.dt_max", f"win{w}.byte_rate"]
    return np.column_stack([c.to_numpy() for c in cols]), names


PROTO_WINDOW_COLS = ["is_tcp", "is_udp", "is_icmp", "syn", "ack", "fin", "rst", "psh", "payload_len", "web_port"]


def proto_window_features(P, cols, stream, sizes):
    """Causal window means of per-packet protocol fields (share of TCP/UDP/ICMP packets, of SYN/ACK/FIN/RST/PSH
    flags, mean payload) over the previous W packets of the same stream (min_periods 1, as window_features)."""
    n = len(P)
    idx = np.arange(n)
    starts = np.flatnonzero(np.r_[True, stream[1:] != stream[:-1]])
    gs = starts[np.searchsorted(starts, idx, "right") - 1]
    out, names = [], []
    for w in sizes:
        s = np.maximum(idx - w + 1, gs)
        for c in PROTO_WINDOW_COLS:
            cs = np.concatenate([[0.0], np.cumsum(P[:, cols.index(c)], dtype=np.float64)])
            out.append(((cs[idx + 1] - cs[s]) / (idx - s + 1)).astype(np.float32))
            names.append(f"pwin{w}.{c}")
    return np.column_stack(out), names


def _column_to_float(s: pd.Series) -> np.ndarray:
    """Numeric parse with hex support ('0x0000'); anything else (IPs, MACs) becomes 0."""
    s = s.fillna("").astype(str).str.strip()
    out = pd.to_numeric(s, errors="coerce")
    is_hex = out.isna() & s.str.fullmatch(r"0[xX][0-9a-fA-F]+", na=False)
    if is_hex.any():
        out = out.copy()
        out[is_hex] = s[is_hex].map(lambda v: float(int(v, 16)))
    return out.fillna(0.0).to_numpy(dtype=np.float64)


def _hash(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def load_raw(csv_path: str, cache_dir: str):
    st = os.stat(csv_path)
    key = _hash([os.path.abspath(csv_path), st.st_size, st.st_mtime, RAW_CACHE_VERSION])
    cache = os.path.join(cache_dir, f"uavnidd_raw_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["labels"].astype(str), [str(f) for f in z["features"]]
    df = pd.read_csv(csv_path, dtype=str, encoding="latin-1", low_memory=False)
    df = df[df[LABEL_COL].notna()]
    labels = df[LABEL_COL].str.strip().to_numpy(dtype=str)
    feats = [c for c in df.columns if c != LABEL_COL]
    X = np.column_stack([_column_to_float(df[c]) for c in feats])
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{cache}.{os.getpid()}.tmp.npz"  # written whole, then swapped in: a stopped run never leaves a broken cache
    np.savez_compressed(tmp, X=X, labels=labels, features=np.array(feats))
    os.replace(tmp, cache)
    return X, labels, feats


def build_clean(cfg: dict, root: str, log=print) -> pd.DataFrame:
    dc = cfg["data"]
    out_dir = os.path.join(root, dc["processed_dir"])
    csv_out = os.path.join(out_dir, "uavnidd_clean.csv")
    report_path = os.path.join(out_dir, "clean_report.json")
    key = _hash({"v": 8, **{k: dc.get(k) for k in ("csv_path", "classes", "other_attacks_as", "drop_columns", "drop_duplicates",
                                         "window_sizes", "drop_capture_settings", "feature_set", "split_by_capture", "stream_by_capture",
                                         "rolling_folds", "proto_path")}})
    proto = dc.get("proto_path")  # optional per-packet protocol fields, row-aligned with the raw packet table
    if proto:  # table written as parquet only (a 70+ column CSV of 27 M rows is slow and not needed)
        pq_out = csv_out[:-4] + ".parquet"
        if os.path.exists(report_path) and json.load(open(report_path)).get("key") == key and os.path.exists(pq_out):
            return pd.read_parquet(pq_out)
    report = None
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    if report and report.get("key") == key and os.path.exists(csv_out):
        pq = csv_out[:-4] + ".parquet"  # same table, binary copy: reads in seconds for large datasets (ISOT)
        if os.path.exists(pq) and os.path.getmtime(pq) >= os.path.getmtime(csv_out):
            return pd.read_parquet(pq)
        df = pd.read_csv(csv_out, engine="pyarrow")  # multi-threaded parser, same values
        df.to_parquet(pq, index=False)
        return df

    log("building cleaned dataset (first run only)")
    X, labels, feats = load_raw(dc["csv_path"], os.path.join(root, "data_cache"))
    order = np.arange(len(labels))
    # Capture files, found without labels: packet numbers continue +1 inside a capture (computed on the raw
    # stream, before any row is removed, so removed rows don't look like capture breaks).
    capture = np.cumsum(np.concatenate([[True], np.diff(X[:, feats.index("frame.number")]) != 1])) - 1
    size_timing = dc.get("feature_set") == "size_timing"
    if size_timing and dc["drop_duplicates"]:
        # Remove only true duplicates: the same packet (same label, capture position, size and gap) stored twice,
        # e.g. UAV-NIDD's DoS capture appears twice. Repeated traffic later in time (periodic telemetry, floods) is
        # real and stays. Done before the time segments and windows, which then see the real packet stream.
        ident = pd.DataFrame({"l": labels, "n": X[:, feats.index("frame.number")], "len": X[:, feats.index("frame.len")],
                              "dt": X[:, feats.index("frame.time_delta_displayed")]})
        keep = ~ident.duplicated().to_numpy()
        X, labels, order, capture = X[keep], labels[keep], order[keep], capture[keep]
    fn = X[:, feats.index("frame.number")]
    rolling = dc.get("rolling_folds", False)
    segment = (time_segments(labels, dc["test_frac"], dc["val_frac"],
                             fn if dc.get("split_by_capture") else None)
               if dc.get("split") == "temporal" and not rolling else None)
    raw_feats = list(feats)
    if dc.get("window_sizes"):
        # Window streams: by capture (as a monitor sees one recording), or by label run (older setups).
        stream_key = capture if dc.get("stream_by_capture") else labels
        W, wnames = window_features(X, stream_key, feats, dc["window_sizes"], segment, full=size_timing)
        X, feats = np.hstack([X, W]), feats + wnames
    raw_counts = pd.Series(labels).value_counts().to_dict()
    source = labels.copy()
    if dc.get("other_attacks_as"):
        labels = np.where(np.isin(labels, dc["classes"] + [dc["normal_class"]]), labels, dc["other_attacks_as"])
        classes = list(dc["classes"]) + [dc["other_attacks_as"]]
    else:
        classes = list(dc["classes"])
    keep_rows = np.isin(labels, classes)
    drop = [c for c in dc["drop_columns"] + dc.get("drop_capture_settings", []) if c in feats]
    keep_cols = [i for i, c in enumerate(feats) if c not in drop]
    df = pd.DataFrame(X[keep_rows][:, keep_cols], columns=[feats[i] for i in keep_cols])
    df[LABEL_COL] = labels[keep_rows]
    df[ORDER_COL] = order[keep_rows]
    df[SOURCE_COL] = source[keep_rows]
    df[CAPTURE_COL] = capture[keep_rows]
    if segment is not None:
        df[SEGMENT_COL] = segment[keep_rows]
    before = df[LABEL_COL].value_counts().to_dict()
    if dc["drop_duplicates"] and not size_timing:  # (size_timing: true duplicates already removed above)
        df = df.drop_duplicates(subset=[c for c in df.columns if c not in (ORDER_COL, SOURCE_COL, SEGMENT_COL, CAPTURE_COL)]).reset_index(drop=True)
    if size_timing:
        # Behaviour-only inputs (valid on any capture, even encrypted): frame length, inter-arrival time and their
        # window statistics. Every header field is dropped: UAV-NIDD shifts header columns between capture sessions,
        # so their values identify the recording rather than the attack.
        model_cols = ["frame.len", "frame.time_delta_displayed"] + [c for c in df.columns if c.startswith("win")]
        df = df[model_cols + [c for c in (LABEL_COL, ORDER_COL, SOURCE_COL, SEGMENT_COL, CAPTURE_COL) if c in df.columns]]
    if proto:
        # Per-packet protocol fields (tools/isot_proto_extract.py: IP/ARP, TCP/UDP/ICMP, TCP flags, payload length,
        # web port; no addresses, counters or clocks) and their window means over the same capture streams.
        ppath = proto if os.path.isabs(proto) else os.path.join(root, proto)
        pcols = json.load(open(ppath[:-4] + "_cols.json"))
        P = np.asarray(np.load(ppath, mmap_mode="r")[df[ORDER_COL].to_numpy()], dtype=np.float32)
        PW, pwn = proto_window_features(P, pcols, df[CAPTURE_COL].to_numpy(), dc["window_sizes"])
        meta = [c for c in (LABEL_COL, ORDER_COL, SOURCE_COL, SEGMENT_COL, CAPTURE_COL) if c in df.columns]
        base = [c for c in df.columns if c not in meta]
        extra = pd.DataFrame(np.hstack([P, PW]), columns=[f"proto.{c}" for c in pcols] + pwn, index=df.index)
        df = pd.concat([df[base], extra, df[meta]], axis=1)
        del P, PW, extra
    after = df[LABEL_COL].value_counts().to_dict()

    os.makedirs(out_dir, exist_ok=True)
    if not proto:
        df.to_csv(csv_out, index=False)
    df.to_parquet(csv_out[:-4] + ".parquet", index=False)
    report = {
        "key": key, "source": dc["csv_path"], "classes": classes,
        "dropped_columns": drop,
        "dropped_columns_reason": "clock / counter / address content (in at least one capture session): "
                                  "identifies when or where a packet was captured, not how the traffic behaves",
        "dropped_capture_settings": dc.get("drop_capture_settings", []),
        "window_sizes": dc.get("window_sizes"),
        "kept_columns": [c for c in df.columns if c not in (LABEL_COL, ORDER_COL, SOURCE_COL, SEGMENT_COL, CAPTURE_COL)],
        "feature_set": dc.get("feature_set", "all"),
        "order_column": f"{ORDER_COL}: raw row index (capture order), used only for the time-based split",
        "raw_label_counts": raw_counts, "rows_before_dedup": before, "rows_after_dedup": after,
        "balancing": "no rows removed for balance; each drone uses a class-weighted loss on its own data; "
                     "validation and test keep the natural class mix",
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"clean dataset: {len(df)} rows -> {csv_out}")
    return df


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class Client:
    cid: int
    role: str  # "general" | "rare" | "poisoned"
    X_train: np.ndarray
    y_train: np.ndarray        # training labels as the drone sees them (flipped for the poisoned drone)
    X_test: np.ndarray         # local test rows + public exam (voting set)
    y_test: np.ndarray
    counts: dict               # true-label counts of the drone's own rows
    vote_groups: np.ndarray = None  # burst (episode) id per voting row; rows of one burst are not independent


@dataclass
class FedData:
    class_names: list
    feature_names: list
    clients: list
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_public: np.ndarray
    y_public: np.ndarray
    rare_target: int
    normal_target: int
    manifest: dict
    scaler: object = None  # fitted on the public exam; reused to prepare new traffic (mixed-traffic test)
    X_test_pure: np.ndarray = None  # mixed-traffic setups: the per-capture (pure) test, kept as a clean reference
    y_test_pure: np.ndarray = None
    test_events: dict = None        # mixed-traffic test: episode id, attack-packet flag and time per packet

    @property
    def n_features(self):
        return self.X_val.shape[1]


def flip_labels(y, mode, normal_target, n_classes):
    if mode == "none":
        return y
    if mode == "attack_to_normal":
        return np.full_like(y, normal_target)
    if mode == "rotate":
        return (y + 1) % n_classes
    raise ValueError(f"unknown flip_mode {mode!r}")


def _split(idx, strat, frac, seed):
    try:
        return train_test_split(idx, test_size=frac, random_state=seed, stratify=strat)
    except ValueError:
        return train_test_split(idx, test_size=frac, random_state=seed)


def _dirichlet(idx_by_class, n_clients, alpha, rng, min_size, floor=0.0):
    """Label-skewed split. floor mixes each class's Dirichlet proportions with an even split,
    p = (1 - floor) * Dir(alpha) + floor / n, so every drone holds every common class (Jiang Sec. 5.1:
    common data is owned by all participants, only the quantities differ)."""
    for _ in range(500):
        parts = [[] for _ in range(n_clients)]
        for idx in idx_by_class.values():
            idx = rng.permutation(idx)
            p = (1 - floor) * rng.dirichlet([alpha] * n_clients) + floor / n_clients
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p, dtype=int) for p in parts]
    raise RuntimeError("could not satisfy min_client_samples; lower it or raise dirichlet_alpha")


def partition_cached(cfg: dict, seed: int, root: str) -> bool:
    """True when this (data + swarm settings, seed) partition is already in data_cache (no rebuild needed)."""
    st = os.stat(cfg["data"]["csv_path"])
    key = _hash({"v": 2, "data": cfg["data"], "clients": cfg["clients"], "seed": seed,
                 "csv": [os.path.abspath(cfg["data"]["csv_path"]), st.st_size, st.st_mtime]})
    return os.path.exists(os.path.join(root, "data_cache", f"partition_{key}.pkl"))


def shared_dir(cfg: dict, seed: int, root: str) -> str:
    """compute.shared_data: one memory-mappable copy of a seed's partition, read by every job of that seed."""
    st = os.stat(cfg["data"]["csv_path"])
    key = _hash({"v": 2, "data": cfg["data"], "clients": cfg["clients"], "seed": seed,
                 "csv": [os.path.abspath(cfg["data"]["csv_path"]), st.st_size, st.st_mtime],
                 "f16": cfg.get("compute", {}).get("feature_dtype") == "float16"})
    return os.path.join(root, "data_cache", f"shared_{key}")


_FEATURE_FIELDS = {"X_train", "X_test", "X_val", "X_public", "X_test_pure"}


def save_shared(data, d: str, f16: bool):
    """Write a partition once as .npy files + a small skeleton pickle. Feature arrays are stored in float16 when
    compute.feature_dtype = float16 (exactly the cast every job applied before), so jobs read identical values."""
    import pickle, dataclasses, shutil
    tmp = f"{d}.{os.getpid()}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    n = [0]

    def put(name, a):
        if isinstance(a, np.ndarray) and a.nbytes > (1 << 20):
            if f16 and name in _FEATURE_FIELDS and a.dtype != np.float16:
                a = a.astype(np.float16)
            fn = f"a{n[0]:04d}.npy"; n[0] += 1
            np.save(os.path.join(tmp, fn), a)
            return ("__npy__", fn)
        return a

    def strip(obj):
        upd = {f.name: put(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)
               if isinstance(getattr(obj, f.name), np.ndarray)}
        return dataclasses.replace(obj, **upd)

    skel = strip(data)
    skel = dataclasses.replace(skel, clients=[strip(c) for c in data.clients])
    if isinstance(data.test_events, dict):
        skel = dataclasses.replace(skel, test_events={k: put(k, v) for k, v in data.test_events.items()})
    with open(os.path.join(tmp, "skeleton.pkl"), "wb") as f:
        pickle.dump(skel, f, protocol=pickle.HIGHEST_PROTOCOL)
    if os.path.exists(d):
        shutil.rmtree(tmp)  # another process finished first
    else:
        os.replace(tmp, d)


def load_shared(d: str):
    """Partition whose large arrays are read-only memory maps: all jobs of a seed share one copy in RAM."""
    import pickle, dataclasses
    with open(os.path.join(d, "skeleton.pkl"), "rb") as f:
        skel = pickle.load(f)
    get = lambda a: np.load(os.path.join(d, a[1]), mmap_mode="r") if isinstance(a, tuple) and a[:1] == ("__npy__",) else a

    def fill(obj):
        return dataclasses.replace(obj, **{f.name: get(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                                           if isinstance(getattr(obj, f.name), tuple)})

    data = fill(skel)
    data = dataclasses.replace(data, clients=[fill(c) for c in skel.clients])
    if isinstance(skel.test_events, dict):
        data = dataclasses.replace(data, test_events={k: get(v) for k, v in skel.test_events.items()})
    return data


def make_partition(cfg: dict, seed: int, root: str, log=print, save_files=True) -> FedData:
    """Cached per (data + swarm settings, seed): built once, then reused by every job and run."""
    import pickle
    st = os.stat(cfg["data"]["csv_path"])  # the data file's identity too: a rebuilt file must not reuse old partitions
    key = _hash({"v": 2, "data": cfg["data"], "clients": cfg["clients"], "seed": seed,
                 "csv": [os.path.abspath(cfg["data"]["csv_path"]), st.st_size, st.st_mtime]})
    cache = os.path.join(root, "data_cache", f"partition_{key}.pkl")
    if os.path.exists(cache) and not save_files:
        with open(cache, "rb") as f:
            return pickle.load(f)
    data = _make_partition(cfg, seed, root, log, save_files)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = f"{cache}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache)
    return data


def _make_partition(cfg: dict, seed: int, root: str, log=print, save_files=True, extras=None) -> FedData:
    """extras (opt-in, evaluation only): a dict that receives the exact split report and the out-of-time slice, and
    the function returns right after the split (no mixing, no FedData); training never passes it."""
    dc, cc = cfg["data"], cfg["clients"]
    future_idx = np.array([], dtype=int)
    df = build_clean(cfg, root, log)
    feats = [c for c in df.columns if c not in (LABEL_COL, ORDER_COL, SOURCE_COL, SEGMENT_COL, CAPTURE_COL)]
    labels = df[LABEL_COL].to_numpy(dtype=str)
    classes = list(dc["classes"]) + ([dc["other_attacks_as"]] if dc.get("other_attacks_as") else [])
    y_all = np.array([classes.index(l) for l in labels])
    rare_t, normal_t = classes.index(dc["rare_class"]), classes.index(dc["normal_class"])
    rng = np.random.default_rng(seed)

    # Held-out splits (stratified, natural class mix): test for final reporting, val for tuning/curves.
    idx = np.arange(len(df))
    if dc.get("split", "random") == "temporal":
        # Time-based hold-out per class: earliest packets -> drones, then validation, latest -> test. Every
        # test packet comes after every training packet of its capture, as in a deployed IDS.
        # Split per original attack type: "Other attack" merges six attacks recorded one after another, so a split
        # on the merged label would put whole attacks (e.g. Jamming) only in the test set.
        if dc.get("rolling_folds"):
            seg = fold_segments(df, seed % 5, dc["test_frac"], dc["val_frac"], dc.get("rotate_only"),
                                dc.get("attacks_per_capture", False))
            pool, val_idx, test_idx = idx[seg == 0], idx[seg == 1], idx[seg == 2]
            future_idx = idx[seg == -1]  # later than this fold's test block: never trained or tested on
        elif SEGMENT_COL in df.columns:  # segments fixed on the raw capture (windows were reset at these boundaries)
            seg = df[SEGMENT_COL].to_numpy()
            pool, val_idx, test_idx = idx[seg == 0], idx[seg == 1], idx[seg == 2]
        else:
            order = df[ORDER_COL].to_numpy() if ORDER_COL in df.columns else idx
            src = df[SOURCE_COL].to_numpy(dtype=str) if SOURCE_COL in df.columns else labels
            pool, val_idx, test_idx = [], [], []
            for c in np.unique(src):
                rows = idx[src == c][np.argsort(order[src == c], kind="stable")]
                n_te, n_va = int(round(len(rows) * dc["test_frac"])), int(round(len(rows) * dc["val_frac"]))
                pool.append(rows[:len(rows) - n_te - n_va])
                val_idx.append(rows[len(rows) - n_te - n_va:len(rows) - n_te])
                test_idx.append(rows[len(rows) - n_te:])
            pool, val_idx, test_idx = (np.sort(np.concatenate(a)) for a in (pool, val_idx, test_idx))
    else:
        rest, test_idx = _split(idx, labels, dc["test_frac"], seed)
        pool, val_idx = _split(rest, labels[rest], dc["val_frac"] / (1 - dc["test_frac"]), seed)
    # Public exam: small class-balanced third-party sample (no drone trains on it).
    public_idx = np.concatenate([rng.choice(pool[labels[pool] == c], min(dc["public_per_class"],
                                            int((labels[pool] == c).sum()) // 4), replace=False)
                                 for c in classes])
    pool = np.setdiff1d(pool, public_idx)

    n = cc["num_clients"]
    perm = rng.permutation(n)
    rare_ids = sorted(perm[:cc["num_rare"]].tolist())
    poison_ids = sorted(perm[cc["num_rare"]:cc["num_rare"] + cc["num_poisoned"]].tolist())
    common = {c: pool[labels[pool] == c] for c in classes if c != dc["rare_class"] or not rare_ids}
    parts = _dirichlet(common, n, cc["dirichlet_alpha"], rng, cc["min_client_samples"],
                       cc.get("class_floor", 0.0))
    if rare_ids:
        rare_rows = rng.permutation(pool[labels[pool] == dc["rare_class"]])
        # Stress test only: the rare drone saw just a fraction of the rare traffic (an attack seen briefly).
        rare_rows = rare_rows[:max(1, int(round(len(rare_rows) * cc.get("rare_fraction", 1.0))))]
        for k, chunk in enumerate(np.array_split(rare_rows, len(rare_ids))):
            parts[rare_ids[k]] = np.concatenate([parts[rare_ids[k]], chunk])

    # Shared feature scaling fitted on the public exam only (no client data is pooled).
    Xf = signed_log(df[feats].to_numpy(dtype=np.float64))
    scaler = StandardScaler().fit(Xf[public_idx])
    Xs = np.clip(scaler.transform(Xf), -10, 10).astype(np.float32)
    # Frequent-value indicators: UAV-NIDD's shifted columns often hold codes (flags, rates, subtypes), which a
    # log-scaled number blurs together. Each column's most frequent values in the public exam become 0/1 inputs,
    # so the vocabulary also comes from the public exam only.
    vi = dc.get("value_indicators") or {}
    vocab = []
    if vi.get("top_k", 0) > 0:
        raw = df[feats].to_numpy(dtype=np.float64)
        for j, c in enumerate(feats):
            vals, cnt = np.unique(raw[public_idx, j], return_counts=True)
            order = np.argsort(-cnt, kind="stable")
            vocab += [(j, c, float(vals[k])) for k in order[:vi["top_k"]] if cnt[k] >= vi.get("min_count", 3)]
        Xs = np.hstack([Xs, np.column_stack([raw[:, j] == v for j, _, v in vocab]).astype(np.float32)])
    feats_in = feats + [f"{c}=={v:g}" for _, c, v in vocab]

    if extras is not None:  # evaluation extras: exact split counts + the out-of-time slice (same features/scaling)
        cnt = lambda rows: {c: int((labels[rows] == c).sum()) for c in classes}
        drones = []
        for cid in range(n):
            rows = parts[cid]
            tr, te = _split(rows, labels[rows], dc["client_test_frac"], seed + cid)  # as in the client loop below
            drones.append({"cid": cid, "role": "rare" if cid in rare_ids else "poisoned" if cid in poison_ids else "general",
                           "train": cnt(tr), "vote": cnt(te)})
        extras["report"] = {"seed": seed, "fold": seed % 5 if dc.get("rolling_folds") else None, "classes": classes,
                            "total": cnt(idx), "drone_pool": cnt(pool), "public_exam": cnt(public_idx),
                            "validation": cnt(val_idx), "test": cnt(test_idx), "out_of_time": cnt(future_idx),
                            "drones": drones, "vote_with_public": bool(dc.get("vote_with_public", True))}
        extras["test_pure"] = (Xs[test_idx], y_all[test_idx])
        extras["future"] = (Xs[future_idx], y_all[future_idx])
        extras["classes"], extras["normal_t"] = classes, normal_t
        return None

    seed_dir = os.path.join(root, dc["processed_dir"], f"seed_{seed}")
    if save_files:
        os.makedirs(seed_dir, exist_ok=True)

    def save(name, rows, train_labels=None):
        if not save_files:
            return
        out = df.iloc[rows].copy()
        if train_labels is not None:
            out["TrainLabel"] = [classes[t] for t in train_labels]
        out.to_parquet(os.path.join(seed_dir, f"{name}.parquet"), index=False)

    # Mixed traffic (data.traffic = mixed): every split is rebuilt as normal traffic with attack bursts interleaved
    # in time, from that split's own rows only (fldos/traffic.py). The pure per-capture test is kept as a reference.
    mixed = dc.get("traffic") == "mixed"
    if mixed:
        from .traffic import build as build_mixed
        base_len = df["frame.len"].to_numpy(float)
        base_gap = df["frame.time_delta_displayed"].to_numpy(float)
        base_order = df[ORDER_COL].to_numpy()
        mrng = np.random.default_rng(seed + 12345)
        # data.proto_path: per-packet protocol fields travel with their packets into the mixed stream and their
        # window means are recomputed there (same column order as the cleaned table)
        pc = [c for c in feats if c.startswith("proto.")]
        base_extra = df[pc].to_numpy(np.float32) if pc else None
        ex = (lambda r: {"extra": base_extra[r], "extra_cols": [c[6:] for c in pc]}) if pc else (lambda r: {})

        def mix(rows):
            Xr, y, info = build_mixed(base_len[rows], base_gap[rows], y_all[rows], base_order[rows], normal_t, mrng,
                                      dc["window_sizes"], **dc.get("mixed", {}), **ex(rows))
            return np.clip(scaler.transform(signed_log(Xr)), -10, 10).astype(np.float32), y, info

    # Public exam (third-party sample) as voters and the server see it. data.public_mixed: built as mixed traffic
    # like every other split (own rng, so the drones' and test traffic are unchanged); otherwise pure capture windows.
    X_pub, y_pub = Xs[public_idx], y_all[public_idx]
    if mixed and dc.get("public_mixed"):
        Xr, y_pub, _ = build_mixed(base_len[public_idx], base_gap[public_idx], y_all[public_idx], base_order[public_idx],
                                   normal_t, np.random.default_rng(seed + 54321), dc["window_sizes"], **dc.get("mixed", {}),
                                   **ex(public_idx))
        X_pub = np.clip(scaler.transform(signed_log(Xr)), -10, 10).astype(np.float32)

    clients = []
    for cid in range(n):
        rows = parts[cid]
        role = "rare" if cid in rare_ids else "poisoned" if cid in poison_ids else "general"
        tr, te = _split(rows, labels[rows], dc["client_test_frac"], seed + cid)
        if mixed:
            X_tr, y_tr, _ = mix(tr)
        else:
            X_tr, y_tr = Xs[tr], y_all[tr]
        if role == "poisoned":
            y_tr = flip_labels(y_tr, cc["flip_mode"], normal_t, len(classes))
        # Voting set: the drone's own local test rows (Jiang Eq. 7-8), optionally topped up with the public exam.
        vote_public = dc.get("vote_with_public", True)
        if mixed:
            X_te, y_te, te_info = mix(te)
            grp = te_info["episode"]
        else:
            X_te, y_te = Xs[te], y_all[te]
            grp = np.arange(len(y_te))
        cap = dc.get("vote_cap")
        if cap and len(y_te) > cap:  # each voter scores on a fixed random sample of its own traffic (speed)
            keep = np.sort(np.random.default_rng(seed * 101 + cid).choice(len(y_te), cap, replace=False))
            X_te, y_te, grp = X_te[keep], y_te[keep], grp[keep]
        if vote_public:  # public exam rows are an independent random sample: one group each
            X_te, y_te = np.concatenate([X_te, X_pub]), np.concatenate([y_te, y_pub])
            grp = np.concatenate([grp, grp.max(initial=-1) + 1 + np.arange(len(y_pub))])
        counts = {c: int((labels[rows] == c).sum()) for c in classes}
        clients.append(Client(cid, role, X_tr, y_tr, X_te, y_te, counts, grp))
        # saved files hold the drone's original rows; TrainLabel = the label it trains with (flipped if poisoned)
        y_rows = flip_labels(y_all[tr], cc["flip_mode"], normal_t, len(classes)) if role == "poisoned" else y_all[tr]
        save(f"drone_{cid + 1:02d}_train", tr, y_rows)
        save(f"drone_{cid + 1:02d}_test", te)
        if save_files:
            log(f"seed {seed} drone {cid + 1:02d} [{role:8s}] train={len(tr):6d} " +
                " ".join(f"{c}={v}" for c, v in counts.items()))
    save("validation", val_idx)
    save("test", test_idx)
    save("public_exam", public_idx)

    manifest = {
        "seed": seed, "classes": classes, "features": feats,
        # Typical traffic mix (class frequencies of the training pool): lets the server score the balanced public
        # exam as if it were natural traffic, so false alarms on big classes cost what they cost in deployment.
        "class_prior": (np.bincount(y_all[pool], minlength=len(classes)) / len(pool)).tolist(),
        "roles": {f"drone_{c.cid + 1:02d}": c.role for c in clients},
        "drone_counts": {f"drone_{c.cid + 1:02d}": c.counts for c in clients},
        "n_val": len(val_idx), "n_test": len(test_idx), "n_public": len(public_idx),
        "public_counts": pd.Series(labels[public_idx]).value_counts().to_dict(),
        "test_counts": pd.Series(labels[test_idx]).value_counts().to_dict(),
        "notes": "drone_XX_train has TrainLabel (flipped for the poisoned drone); each drone's voting set is "
                 "drone_XX_test + public_exam; scaling = signed log1p then StandardScaler fitted on public_exam",
    }
    if save_files:
        with open(os.path.join(seed_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    if mixed:
        X_val, y_val, _ = mix(val_idx)
        X_test, y_test, test_info = mix(test_idx)
        manifest.update(traffic="mixed", n_val_mixed=int(len(y_val)), n_test_mixed=int(len(y_test)))
        return FedData(classes, feats_in, clients, X_val, y_val, X_test, y_test, X_pub, y_pub,
                       rare_t, normal_t, manifest, scaler=scaler, X_test_pure=Xs[test_idx], y_test_pure=y_all[test_idx],
                       test_events=test_info)
    return FedData(classes, feats_in, clients, Xs[val_idx], y_all[val_idx], Xs[test_idx], y_all[test_idx],
                   Xs[public_idx], y_all[public_idx], rare_t, normal_t, manifest, scaler=scaler)
