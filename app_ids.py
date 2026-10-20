"""IDS-console additions to the FL DoS Swarm dashboard (app.py). Read-only: no training, nothing is written to runs/,
data/, data_cache/ or dqn_store/.

  Test          score the saved global model(s) on the test stream on command (timed), metrics + confusion matrix
  Live          replay of the test stream packet by packet, as an IDS sensor would see it (play / pause / step / speed,
                compare methods, jump to a real attack burst), running KPIs and confusion matrix
  Alerts        alert console: flagged packets grouped into alerts (k of last m), severity, ground-truth verdict,
                analyst triage (kept in the browser session only), IDS KPIs (detection rate, false alarms / hour, delay)
  Verification  unit tests, goal checks and trend checks run on command (subprocess, read-only)
  + status bar, CSS layer, and "Why was this drone dropped / readmitted?" inside the Swarm monitor.

Every dataset-specific name (classes, rare class, DoS classes) is read from the run's meta.json.
"""
import html
import json
import os
import pickle
import subprocess
import sys
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import app_roles

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
PY = os.path.join(ROOT, ".venv", "bin", "python")

LABELS = {}
_PURPOSE = [lambda t: note(t)]


def configure(method_labels, purpose_fn=None):
    """Called once by app.py: reuse its method names and its grey purpose() text."""
    LABELS.update(method_labels)
    if purpose_fn is not None:
        _PURPOSE[0] = purpose_fn


def lab(m):
    return LABELS.get(m, m)


def purpose(text):
    _PURPOSE[0](text)


def note(text):
    """Explanatory grey text, shown only when 'Show explanations' is on in the sidebar."""
    if st.session_state.get("show_notes", False):
        st.caption(app_roles.mark(text))


def state_badge(state):
    """Run state as a small coloured dot + uppercase label (no emoji)."""
    color = {"done": "var(--ids-ok)", "running": "var(--ids-warn)", "preparing": "var(--ids-warn)",
             "failed": "var(--ids-bad)"}.get(state, "rgba(128,128,128,.8)")
    text = {"running": "training", "preparing": "preparing data"}.get(state, state)
    return f'<span class="ids-state"><span class="ids-dot" style="background:{color}"></span>{esc(text)}</span>'


def esc(x):
    return html.escape(str(x))


# Severity scale (Elastic Security's four levels: critical = likely incident, high = immediate triage,
# medium = needs investigation, low = of interest).
SEV_ORDER = ["critical", "high", "medium", "low"]
SEV_COLORS = {"critical": "#bd271e", "high": "#e7664c", "medium": "#d6bf57", "low": "#54b399"}
SEV_TEXT = {"critical": "#ffffff", "high": "#ffffff", "medium": "#1a1c21", "low": "#1a1c21"}
IDLE_CAP = 10.0          # s: idle gaps longer than this are shortened on the stream clock (tcpreplay --maxsleep idea)
REPLAY_SLEEP_CAP = 1.0   # s: "real time" replay never waits longer than this between two packets
LOW_CONF = 0.5           # alerts whose strongest packet is below this confidence are "low"

CSS = """
:root{--ids-mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
--ids-border:rgba(128,128,128,.30);--ids-fill:rgba(128,128,128,.07);--ids-ok:#3f9f4f;--ids-bad:#d64545;--ids-warn:#c99a06}
.block-container{padding-top:3.2rem;padding-bottom:2rem}
[data-testid="stAppDeployButton"],[data-testid="stMainMenu"] + div:empty{display:none}
.ids-state{font-family:var(--ids-mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;opacity:.85}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{white-space:normal!important;overflow:visible!important;text-overflow:clip!important}
.ids-scroll{overflow-x:auto;max-width:100%}
[data-baseweb="select"] [data-baseweb="tag"]{max-width:none}
div[data-testid="stMultiSelect"] [data-baseweb="select"]>div{flex-wrap:wrap}
div[data-testid="stMetric"]{border:1px solid var(--ids-border);border-radius:3px;padding:6px 10px;background:var(--ids-fill)}
div[data-testid="stMetricValue"]{font-family:var(--ids-mono);font-size:1.3rem}
[data-testid="stMetricLabel"] p{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--ids-border);flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{padding:4px 12px;font-size:.84rem}
.stTabs [data-baseweb="tab"] p{font-size:.84rem;font-weight:600;letter-spacing:.01em}
div[data-testid="stExpander"] details{border-radius:3px}
.ids-status{font-family:var(--ids-mono);font-size:.74rem;border:1px solid var(--ids-border);background:var(--ids-fill);
padding:4px 10px;border-radius:3px;display:flex;flex-wrap:wrap;gap:2px 16px;margin:0 0 6px 0;line-height:1.6}
.ids-status span.k{opacity:.65;text-transform:uppercase;font-size:.66rem;letter-spacing:.05em;margin-right:4px}
.ids-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}
.ids-sev{display:inline-block;min-width:58px;text-align:center;padding:0 6px;border-radius:2px;font-size:.68rem;
font-weight:700;text-transform:uppercase;font-family:var(--ids-mono);letter-spacing:.03em}
.ids-banner{font-family:var(--ids-mono);font-size:.80rem;padding:6px 10px;border:1px solid var(--ids-border);
border-left-width:5px;border-radius:3px;margin:4px 0 8px 0;background:var(--ids-fill)}
.ids-kv{font-family:var(--ids-mono);font-size:.76rem;opacity:.9}
table.ids{border-collapse:collapse;width:100%;font-family:var(--ids-mono);font-size:.74rem;margin-bottom:6px}
table.ids th{text-align:left;font-weight:600;border-bottom:1px solid var(--ids-border);padding:2px 6px;
text-transform:uppercase;font-size:.64rem;letter-spacing:.04em;opacity:.7;white-space:nowrap}
table.ids td{padding:1px 6px;border-bottom:1px solid rgba(128,128,128,.13);white-space:nowrap}
table.ids td.r,table.ids th.r{text-align:right}
table.ids tr.fa td{background:rgba(214,191,87,.20)}
table.ids tr.miss td{background:rgba(189,39,30,.18)}
table.ids tr.tp td{background:rgba(231,102,76,.09)}
table.ids tr.wt td{background:rgba(231,102,76,.16)}
table.ids td.fa{background:rgba(214,191,87,.25)} table.ids td.miss{background:rgba(189,39,30,.22)}
table.ids td.tp{background:rgba(231,102,76,.10)} table.ids td.wt{background:rgba(231,102,76,.18)}
.ids-out{font-family:var(--ids-mono);font-size:.72rem;line-height:1.35;white-space:pre-wrap;border:1px solid var(--ids-border);
background:var(--ids-fill);padding:6px 8px;border-radius:3px;max-height:460px;overflow:auto}
.ids-out .ok{color:var(--ids-ok);font-weight:700}.ids-out .bad{color:var(--ids-bad);font-weight:700}
.ids-out .dim{opacity:.6}
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    if st.session_state.get("ui_dark"):
        st.markdown(f"<style>{CSS_DARK}</style>", unsafe_allow_html=True)


# Dark mode (toggle in the sidebar): a security-operations-console palette - near-black blue-grey surfaces, raised panels,
# soft off-white text, blue accent; status/severity colours stay the same hues. Applied through Streamlit's own theme
# options, so every native element (tables, inputs, charts' axes and text) switches with it.
THEME_LIGHT = {"base": "light", "primaryColor": "#2563eb", "backgroundColor": "#ffffff",
               "secondaryBackgroundColor": "#f0f2f6", "textColor": "#31333f"}
THEME_DARK = {"base": "dark", "primaryColor": "#4a8cf7", "backgroundColor": "#0e1217",
              "secondaryBackgroundColor": "#161c24", "textColor": "#d5dde6"}
CSS_DARK = """
:root{--ids-border:rgba(150,165,185,.28);--ids-fill:rgba(110,135,170,.09);--ids-ok:#4fbf63;--ids-bad:#ff6b6b;--ids-warn:#e0b43a}
[data-testid="stSidebar"]{border-right:1px solid rgba(150,165,185,.18)}
table.ids td{border-bottom:1px solid rgba(150,165,185,.14)}
"""


def theme_toggle():
    """Sidebar Dark mode switch. First visit follows the browser/OS setting, as before."""
    if "ui_dark" not in st.session_state:
        try:
            st.session_state.ui_dark = st.context.theme.type == "dark"
        except Exception:
            st.session_state.ui_dark = False
    dark = st.sidebar.toggle("Dark mode", key="ui_dark")
    want = THEME_DARK if dark else THEME_LIGHT
    if any(st._config.get_option(f"theme.{k}") != v for k, v in want.items()):
        for k, v in want.items():
            st._config.set_option(f"theme.{k}", v)
        st.rerun()


def ink():
    """Text/rule colour for chart marks that were hard-coded black."""
    return "#d5dde6" if st.session_state.get("ui_dark") else "black"


def sev_badge(sev):
    return (f'<span class="ids-sev" style="background:{SEV_COLORS[sev]};color:{SEV_TEXT[sev]}">{esc(sev)}</span>')


def fmt_clock(sec):
    """Stream clock as +HHH:MM:SS.mmm since the start of the test stream."""
    if sec is None or not np.isfinite(sec):
        return "-"
    h, rem = divmod(float(sec), 3600)
    m, s = divmod(rem, 60)
    return f"+{int(h):03d}:{int(m):02d}:{s:06.3f}"


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


# ---------- status bar ----------
@st.cache_data(ttl=60, show_spinner=False)
def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return ""


def _last_update(run_dir, meta):
    ts = [meta.get("finished") or 0]
    for m in meta.get("methods", []):
        for s in meta.get("seeds", []):
            p = os.path.join(run_dir, m, f"seed_{s}", "summary.json")
            if os.path.exists(p):
                ts.append(os.path.getmtime(p))
    return max(ts) or os.path.getmtime(os.path.join(run_dir, "meta.json"))


def model_version(run_dir, meta):
    """The project's version tag (git tag '*final*') created right after this run's models were written."""
    last = _last_update(run_dir, meta)
    tags = []
    for line in _git("for-each-ref", "--format=%(refname:short) %(creatordate:unix)", "refs/tags").splitlines():
        try:
            name, t = line.split()
            tags.append((name, int(t)))
        except ValueError:
            continue
    prefix = os.path.basename(run_dir).split("-")[0].lower()  # archived copies are named after their version
    named = [n for n, _ in tags if n.lower() == f"{prefix}-final"]
    if named:
        return named[0]
    cand = sorted((t, n) for n, t in tags if "final" in n.lower() and t >= last - 3600)
    return cand[0][1] if cand else "untagged"


def secure_link_label(run_dir, meta):
    """Status-bar text of the run's drone <-> server link: 'mTLS · TLS 1.3 · <cipher>' for real mTLS runs (cipher from
    the newest job's mtls_meta.json), 'on · AES-256-GCM' for older runs with the in-process link, None when the run used neither."""
    import app_mtls  # lazy: app_mtls -> app_roles only, no cycle at import time
    mtls = app_mtls.link_label(run_dir, meta)
    if mtls or not meta.get("secure_link"):
        return mtls
    sl = meta["secure_link"]
    return f"on · {sl.get('aead', 'AES-256-GCM') if isinstance(sl, dict) else 'AES-256-GCM'}"


def status_bar(run, meta, dataset, state):
    run_dir = os.path.join(RUNS, run)
    cfg = meta.get("config", {}) or {}
    cc = cfg.get("clients", {}) or {}
    seeds = meta.get("seeds", [])
    last = _last_update(run_dir, meta)
    color = {"done": "var(--ids-ok)", "running": "var(--ids-warn)", "preparing": "var(--ids-warn)",
             "failed": "var(--ids-bad)"}.get(state, "rgba(128,128,128,.8)")
    rare_n = app_roles.span(f"{cc.get('num_rare', '?')} rare", "rare")
    poison_n = app_roles.span(f"{cc.get('num_poisoned', '?')} poisoned", "poisoned")
    items = [("dataset", dataset), ("run", run),
             ("state", f'<span class="ids-dot" style="background:{color}"></span>{esc(state)}'),
             ("model version", model_version(run_dir, meta)), ("code", _git("log", "-1", "--format=%h") or "-"),
             ("drones", f"{esc(cc.get('num_clients', '?'))} ({rare_n}, {poison_n})"),
             ("seeds", ",".join(map(str, seeds)) or "-"), ("methods", len(meta.get("methods", []))),
             ("rounds", meta.get("rounds", "?")), ("rare class", meta.get("rare_class", "-")),
             ("last update", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)))]
    link = secure_link_label(run_dir, meta)
    if link:  # added after the existing fields: real mTLS runs show the negotiated TLS, older runs their AEAD
        items.insert(-1, ("secure link", link))
    body = "".join(f'<div><span class="k">{esc(k)}</span>{v if k in ("state", "drones") else esc(v)}</div>' for k, v in items)
    st.markdown(f'<div class="ids-status">{body}</div>', unsafe_allow_html=True)


# ---------- test stream (read-only) ----------
class Unavailable(Exception):
    pass


def _meta(run):
    return load_json(os.path.join(RUNS, run, "meta.json"), {}) or {}


@st.cache_resource(max_entries=4, show_spinner="Loading the test stream (first time only)...")
def load_stream(run, seed):
    """The run's test stream for one seed, exactly as used at training time. Read from data_cache/ when the
    partition is cached; otherwise rebuilt in memory from the cleaned dataset (no files written)."""
    meta = _meta(run)
    cfg = meta.get("config")
    if not cfg:
        raise Unavailable(f"runs/{run}/meta.json has no config")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from fldos.data import _hash
    t0 = time.perf_counter()
    dc = cfg["data"]
    if not os.path.exists(dc["csv_path"]):
        raise Unavailable(f"dataset file not found: {dc['csv_path']}")
    stt = os.stat(dc["csv_path"])
    key = _hash({"v": 2, "data": dc, "clients": cfg["clients"], "seed": seed,
                 "csv": [os.path.abspath(dc["csv_path"]), stt.st_size, stt.st_mtime]})
    cache = os.path.join(ROOT, "data_cache", f"partition_{key}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            d = pickle.load(f)
        source = "data_cache (as used in training)"
    else:
        # Rebuild in memory only when the cleaned dataset is already on disk, so that nothing is written.
        ckey = _hash({"v": 8, **{k: dc.get(k) for k in ("csv_path", "classes", "other_attacks_as", "drop_columns",
                                                       "drop_duplicates", "window_sizes", "drop_capture_settings",
                                                       "feature_set", "split_by_capture", "stream_by_capture",
                                                       "rolling_folds")}})
        rep = load_json(os.path.join(ROOT, dc["processed_dir"], "clean_report.json"), {}) or {}
        if rep.get("key") != ckey or not os.path.exists(os.path.join(ROOT, dc["processed_dir"], "uavnidd_clean.csv")):
            raise Unavailable("the test stream for this seed is not cached and the cleaned dataset does not match "
                              "this run's settings; the dashboard does not rebuild data (read-only)")
        from fldos.data import _make_partition
        d = _make_partition(cfg, seed, ROOT, log=lambda *a: None, save_files=False)
        source = "rebuilt in memory from the cleaned dataset (same seed and settings; no files written)"
    X, y = d.X_test, np.asarray(d.y_test, dtype=np.int64)
    n, names, nt = len(y), list(d.class_names), int(d.normal_target)
    feats = list(d.feature_names)

    def raw(col):  # undo the scaling (signed log1p, then standardisation fitted on the public exam)
        if col not in feats or d.scaler is None:
            return None
        j = feats.index(col)
        z = X[:, j].astype(np.float64) * d.scaler.scale_[j] + d.scaler.mean_[j]
        return np.sign(z) * np.expm1(np.abs(z))
    size = raw("frame.len")
    ev = d.test_events
    has_time = ev is not None and all(k in ev for k in ("time", "episode", "attack_packet"))
    if has_time:
        ep = np.asarray(ev["episode"], dtype=np.int64)
        t = np.asarray(ev["time"], dtype=np.float64)
        att = np.asarray(ev["attack_packet"], dtype=bool)
        starts_mask = np.r_[True, np.diff(ep) != 0]
        gap = np.diff(t, prepend=0.0)
        gap[starts_mask] = t[starts_mask]
        gap = np.clip(gap, 0, None)
    else:  # no per-packet timeline: gaps from the scaled feature, one stream
        ep = np.zeros(n, dtype=np.int64)
        att = y != nt
        g = raw("frame.time_delta_displayed")
        gap = np.clip(g, 0, None) if g is not None else np.full(n, 1e-3)
        starts_mask = np.r_[True, np.zeros(n - 1, bool)]
    starts = np.flatnonzero(starts_mask)
    ep_start = starts[np.searchsorted(starts, np.arange(n), side="right") - 1]
    clock = np.cumsum(np.minimum(gap, IDLE_CAP))
    rclock = np.cumsum(np.minimum(gap, REPLAY_SLEEP_CAP))
    bursts = []
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        ai = np.flatnonzero(att[a:b] | (y[a:b] != nt))
        if not len(ai):
            continue
        yy = y[a:b][y[a:b] != nt]
        cls = int(np.bincount(yy, minlength=len(names)).argmax()) if len(yy) else -1
        fa = np.flatnonzero(att[a:b])
        s_idx, e_idx = (a + fa[0], a + fa[-1]) if len(fa) else (a + ai[0], a + ai[-1])
        bursts.append({"episode": int(ep[a]), "cls": cls, "start_idx": int(s_idx), "end_idx": int(e_idx),
                       "start_t": float(clock[s_idx]), "end_t": float(clock[e_idx]), "packets": int(e_idx - s_idx + 1)})
    tc = cfg["training"]
    dos = dc.get("dos_classes") or [c for c in dc["classes"] if c != dc["normal_class"]]
    rare = meta.get("rare_class") or dc.get("rare_class")
    return {"X": X, "y": y, "n": n, "C": len(names), "names": names, "normal": nt,
            "rare": names.index(rare) if rare in names else int(d.rare_target), "rare_name": rare,
            "dos_idx": [names.index(c) for c in dos if c in names], "ep": ep, "ep_start": ep_start, "att": att,
            "gap": gap, "size": size, "clock": clock, "rclock": rclock, "has_time": has_time,
            "bursts": pd.DataFrame(bursts), "n_features": X.shape[1], "hidden": tc.get("hidden", (128, 64)), "dropout": tc.get("dropout", 0.1),
            "load_s": time.perf_counter() - t0, "source": source}


def model_path(run, method, seed):
    return os.path.join(RUNS, run, method, f"seed_{seed}", "global_model.pt")


def score_model(S, run, seed, method):
    """Score the saved global model on the whole test stream (CPU, batches of 65,536 packets), timed."""
    import torch
    from fldos.model import MLP, set_vector
    p = model_path(run, method, seed)
    if not os.path.exists(p):
        raise Unavailable(f"no saved global model for {lab(method)}, seed {seed}")
    g = torch.load(p, map_location="cpu")
    model = MLP(S["n_features"], S["C"], S["hidden"], S["dropout"])
    set_vector(model, g)
    model.eval()
    X = torch.from_numpy(np.ascontiguousarray(S["X"]))
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.cat([torch.softmax(model(X[i:i + 65536]), 1) for i in range(0, len(X), 65536)]).numpy()
    secs = max(time.perf_counter() - t0, 1e-9)
    return {"pred": probs.argmax(1).astype(np.int64), "conf": probs.max(1).astype(np.float32), "secs": secs,
            "pps": len(X) / secs}


@st.cache_resource(max_entries=48, show_spinner=False)
def predictions(run, seed, method):
    """Computed once per (run, seed, method); the Live replay only reveals them over time."""
    return score_model(load_stream(run, seed), run, seed, method)


def ids_metrics(S, pred, y=None):
    y = S["y"] if y is None else y
    C, nt = S["C"], S["normal"]
    cm = np.bincount(y * C + pred, minlength=C * C).reshape(C, C)
    return metrics_from_cm(S, cm)


def metrics_from_cm(S, cm):
    C, nt = S["C"], S["normal"]
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(C), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(C), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(C), where=(prec + rec) > 0)
    att_rows = [i for i in range(C) if i != nt]
    n_att = cm[att_rows].sum()
    dos = S["dos_idx"]
    n_dos = cm[dos].sum()
    tot = cm.sum()
    present = cm.sum(1) > 0
    return {"cm": cm, "prec": prec, "rec": rec, "f1": f1, "support": cm.sum(1),
            "accuracy": tp.sum() / tot if tot else np.nan,
            "macro_f1": f1.mean(),  # same definition as training (every class)
            "macro_f1_present": f1[present].mean() if present.any() else np.nan,
            "dos_detected": (n_dos - cm[dos, nt].sum()) / n_dos if n_dos else np.nan,
            "missed": cm[att_rows, nt].sum() / n_att if n_att else np.nan,
            "false_alarms": (cm[nt].sum() - cm[nt, nt]) / cm[nt].sum() if cm[nt].sum() else np.nan,
            "rare_recall": rec[S["rare"]] if cm[S["rare"]].sum() else np.nan, "rare_f1": f1[S["rare"]]}


def model_methods(run, methods, seeds):
    return [m for m in methods if any(os.path.exists(model_path(run, m, s)) for s in seeds)]


@st.cache_data(ttl=300, show_spinner=False)
def default_method(run, methods, seeds):
    """Default = the Jiang variant with the best VALIDATION macro F1 (final round, mean over seeds); never the test."""
    best, best_v = None, -1.0
    for m in methods:
        if not m.startswith("jiang"):
            continue
        v = [load_json(os.path.join(RUNS, run, m, f"seed_{s}", "summary.json"), {}).get("val_final", {}).get("macro_f1")
             for s in seeds]
        v = [x for x in v if x is not None]
        if v and np.mean(v) > best_v:
            best, best_v = m, float(np.mean(v))
    return best or next((m for m in methods if m.startswith("jiang")), methods[0] if methods else None)


def _pct(x, d=1):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{d}%}"


def _num(x, f="{:.3f}"):
    return "-" if x is None or not np.isfinite(x) else f.format(x)


def cm_chart(cm, names, title, rare=None, normal=None):
    rows = [{"true": names[i], "pred": names[j], "count": int(cm[i, j]), "pct": cm[i, j] / max(1, cm[i].sum()),
             "label": f"{cm[i, j] / max(1, cm[i].sum()):.0%}\n{int(cm[i, j]):,}"}
            for i in range(len(names)) for j in range(len(names))]
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(x=alt.X("pred:N", sort=names, title="Predicted"),
                                y=alt.Y("true:N", sort=names, title="True"))
    heat = base.mark_rect().encode(color=alt.Color("pct:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), legend=None),
                                   tooltip=["true", "pred", "count", alt.Tooltip("pct:Q", format=".1%")])
    txt = base.mark_text(fontSize=9, lineBreak="\n").encode(
        text="label:N", color=alt.condition(alt.datum.pct > 0.5, alt.value("white"), alt.value("black")))
    layers = heat + txt
    if rare in names:
        layers += alt.Chart(pd.DataFrame({"true": [rare]})).mark_rect(fill=None, stroke="#2e7d32", strokeWidth=3) \
            .encode(y=alt.Y("true:N", sort=names))
    if normal in names:
        layers += alt.Chart(pd.DataFrame({"pred": [normal]})).mark_rect(
            fill=None, stroke="#c62828", strokeWidth=2, strokeDash=[4, 2]).encode(x=alt.X("pred:N", sort=names))
    return layers.properties(title=title, height=alt.Step(34), width=alt.Step(44)) \
        .configure_axisX(labelAngle=-45, labelLimit=140, labelOverlap=False)


def html_cm(cm, names):
    """Compact confusion matrix: counts with the row share as cell shading."""
    head = "".join(f'<th class="r">{esc(n[:10])}</th>' for n in names)
    body = []
    for i, n in enumerate(names):
        tot = max(1, cm[i].sum())
        cells = []
        for j in range(len(names)):
            p = cm[i, j] / tot
            col = "66,133,244" if i == j else "214,69,69"
            cells.append(f'<td class="r" style="background:rgba({col},{0.08 + 0.55 * p:.2f})" '
                         f'title="{p:.1%} of true {esc(n)}">{int(cm[i, j]):,}<span style="opacity:.6"> {p:.0%}</span></td>')
        body.append(f'<tr><td>{esc(n[:16])}</td><td class="r">{int(cm[i].sum()):,}</td>{"".join(cells)}</tr>')
    return (f'<div class="ids-scroll"><table class="ids"><tr><th>true \\ pred</th><th class="r">n</th>{head}</tr>{"".join(body)}</table></div>')


def _stream_or_warn(run, seed):
    try:
        return load_stream(run, seed)
    except Unavailable as e:
        st.warning(f"Seed {seed}: {e}.")
    except Exception as e:  # keep the other tabs alive
        st.warning(f"Seed {seed}: could not load the test stream ({type(e).__name__}: {e}).")
    return None


# ---------- Test tab ----------
def tab_test(run_dir, meta, methods, seeds):
    run = os.path.basename(run_dir)
    note("Test on command: load the saved final global model of each FL method and score it on the held-out test "
               "stream now (no training). Timing is measured on this machine.")
    have = model_methods(run, methods, seeds)
    if not have:
        st.info("No saved global models in this run yet (global_model.pt appears when a job finishes).")
        return
    dm = default_method(run, tuple(have), tuple(seeds))
    c1, c2, c3 = st.columns([3, 2, 2])
    method = c1.selectbox("Method", have, index=have.index(dm) if dm in have else 0, format_func=lab, key="ids_test_m")
    seed_opt = c2.selectbox("Seed", ["All seeds"] + list(seeds), key="ids_test_s",
                            format_func=lambda s: s if isinstance(s, str) else f"seed {s}")
    compare = c3.toggle("Compare all methods", False, key="ids_test_cmp",
                        help="Score every method on the same seed(s) and show one table.")
    note("Default method = the Jiang variant with the best validation macro F1 (never chosen on the test set).")
    go = st.button("Run test", type="primary", key="ids_test_go")
    sel_seeds = list(seeds) if seed_opt == "All seeds" else [seed_opt]
    if go:
        todo = have if compare else [method]
        out = {"run": run, "seeds": sel_seeds, "methods": todo, "primary": method, "per": {}, "errors": []}
        t_all = time.perf_counter()
        load_s, prog = 0.0, st.progress(0.0, text="Scoring...")
        for k, s in enumerate(sel_seeds):
            t0 = time.perf_counter()
            S = _stream_or_warn(run, s)
            load_s += time.perf_counter() - t0
            if S is None:
                out["errors"].append(f"seed {s}: test stream unavailable")
                continue
            summ_cache = {}
            for m in todo:
                try:
                    P = score_model(S, run, s, m)
                except Unavailable as e:
                    out["errors"].append(str(e))
                    continue
                M = ids_metrics(S, P["pred"])
                saved = load_json(os.path.join(RUNS, run, m, f"seed_{s}", "summary.json"), {}) or {}
                st_test = saved.get("test", {})
                match = None
                if "accuracy" in st_test and "macro_f1" in st_test:
                    match = abs(st_test["accuracy"] - M["accuracy"]) < 1e-6 and abs(st_test["macro_f1"] - M["macro_f1"]) < 1e-6
                out["per"][(m, s)] = {"M": M, "secs": P["secs"], "n": S["n"], "match": match}
                summ_cache[m] = True
            out["names"], out["rare"], out["normal"] = S["names"], S["rare_name"], S["names"][S["normal"]]
            out["dos_idx"], out["rare_idx"], out["normal_idx"] = S["dos_idx"], S["rare"], S["normal"]
            prog.progress((k + 1) / len(sel_seeds), text=f"Scored seed {s}")
        prog.empty()
        out["load_s"], out["wall_s"] = load_s, time.perf_counter() - t_all
        out["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["ids_test_res"] = out
    res = st.session_state.get("ids_test_res")
    if not res or res.get("run") != run:
        purpose("press Run test to score the saved model(s). Results stay here until the next test.")
        return
    for e in res["errors"]:
        st.warning(e)
    if not res["per"]:
        return
    Sx = {"C": len(res["names"]), "normal": res["normal_idx"], "rare": res["rare_idx"], "dos_idx": res["dos_idx"]}
    n_pk = sum(v["n"] for v in res["per"].values())
    secs = sum(v["secs"] for v in res["per"].values())
    st.markdown(f'<div class="ids-kv">test run {esc(res["when"])} · {len(res["methods"])} method(s) × '
                f'{len(res["seeds"])} seed(s) · scored {n_pk:,} packets in {secs:.3f} s = '
                f'<b>{n_pk / max(secs, 1e-9) / 1e6:.2f} M packets/s</b> (CPU, model inference only) · test-stream load '
                f'{res["load_s"]:.2f} s · total wall time {res["wall_s"]:.2f} s</div>', unsafe_allow_html=True)
    purpose("how fast the global model classifies traffic on this computer, and whether it still scores what it "
            "scored when training finished (match = same accuracy and macro F1 as summary.json for the final model).")

    pm = res["primary"] if res["primary"] in res["methods"] else res["methods"][0]
    show_m = pm
    if len(res["methods"]) > 1:
        rows = []
        for m in res["methods"]:
            per = [res["per"][(m, s)] for s in res["seeds"] if (m, s) in res["per"]]
            if not per:
                continue
            g = lambda k: float(np.nanmean([p["M"][k] for p in per]))
            rows.append({"Method": lab(m), "Accuracy": g("accuracy"), "Macro F1": g("macro_f1"),
                         "DoS detected": g("dos_detected"), "Attacks missed": g("missed"),
                         "False alarms": g("false_alarms"), f"{res['rare']} recall": g("rare_recall"),
                         f"{res['rare']} F1": g("rare_f1"),
                         "Packets/s (M)": sum(p["n"] for p in per) / max(sum(p["secs"] for p in per), 1e-9) / 1e6,
                         "Matches saved score": f"{sum(bool(p['match']) for p in per)}/{len(per)}"})
        st.markdown("#### Comparison: every method on the same test packets")
        df = pd.DataFrame(rows)
        pct_cols = ["Accuracy", "DoS detected", "Attacks missed", "False alarms", f"{res['rare']} recall"]
        st.dataframe(df.style.format({**{c: "{:.1%}" for c in pct_cols}, "Macro F1": "{:.3f}",
                                      f"{res['rare']} F1": "{:.3f}", "Packets/s (M)": "{:.2f}"}),
                     hide_index=True, width="stretch")
        purpose("one row per FL method, same seed(s) and same packets; mean over the selected seeds of the final "
                "global model (the Results tab averages the last 10 models, so values differ slightly).")
        show_m = st.selectbox("Details for", res["methods"], index=res["methods"].index(pm), format_func=lab,
                              key="ids_test_detail")
    per = [res["per"][(show_m, s)] for s in res["seeds"] if (show_m, s) in res["per"]]
    if not per:
        return
    g = lambda k: float(np.nanmean([p["M"][k] for p in per]))
    st.markdown(f"#### {lab(show_m)} · " + ("all seeds (mean)" if len(per) > 1 else f"seed {res['seeds'][0]}"))
    cols = st.columns(7)
    cols[0].metric("Accuracy", _pct(g("accuracy")))
    cols[1].metric("Macro F1", _num(g("macro_f1")))
    cols[2].metric("DoS detected", _pct(g("dos_detected")))
    cols[3].metric("Attacks missed", _pct(g("missed")))
    cols[4].metric("False alarms", _pct(g("false_alarms")))
    cols[5].metric(f"{res['rare']} recall", _pct(g("rare_recall")))
    cols[6].metric(f"{res['rare']} F1", _num(g("rare_f1")))
    matches = [p["match"] for p in per if p["match"] is not None]
    st.caption(f"Matches the score saved at training time: {sum(matches)}/{len(matches)} seed(s). "
               "DoS detected = DoS-family packets flagged as any attack; attacks missed = attack packets called Normal; "
               "false alarms = Normal packets flagged as an attack.")
    cm = sum(p["M"]["cm"] for p in per)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.altair_chart(cm_chart(cm, res["names"], f"Confusion matrix ({len(per)} seed(s) summed): % of row and count",
                                 res["rare"], res["normal"]))
        purpose(f"rows = true class, columns = predicted. Green box = rare class ({res['rare']}); red dashed column = "
                "predicted Normal (for attack rows: missed attacks).")
    with c2:
        M = metrics_from_cm(Sx, cm)
        pc = pd.DataFrame({"Class": res["names"], "Packets": M["support"].astype(int), "Precision": M["prec"],
                           "Recall": M["rec"], "F1": M["f1"],
                           "Called Normal": [cm[i, res["normal_idx"]] / max(1, cm[i].sum()) for i in range(len(res["names"]))],
                           "Type": ["normal" if i == res["normal_idx"] else "rare DoS" if i == res["rare_idx"] else
                                    "DoS" if i in res["dos_idx"] else "other attack" for i in range(len(res["names"]))]})
        st.dataframe(pc.style.format({"Packets": "{:,}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
                                      "Called Normal": "{:.1%}"}), hide_index=True, width="stretch")
        purpose("per-class precision / recall / F1 from the summed matrix; 'Called Normal' for an attack class = its "
                "missed share.")


# ---------- alerts ----------
@st.cache_data(max_entries=32, show_spinner=False)
def build_alerts(run, seed, method, k, m, quiet):
    """Alert = a run of packets where at least k of the last m packets are flagged (per episode); alerts closer than
    `quiet` seconds merge. Returns (alerts, bursts) DataFrames."""
    S, P = load_stream(run, seed), predictions(run, seed, method)
    y, ep, clock, nt, C = S["y"], S["ep"], S["clock"], S["normal"], S["C"]
    pred, conf = P["pred"], P["conf"]
    n = len(y)
    flag = pred != nt
    c = np.r_[0, np.cumsum(flag)]
    i = np.arange(n)
    lo = np.maximum(i - m + 1, S["ep_start"])
    confirmed = (c[i + 1] - c[lo]) >= k
    idx = np.flatnonzero(confirmed)
    truth = (y != nt) | S["att"]
    rows = []
    if len(idx):
        brk = np.r_[True, (np.diff(ep[idx]) != 0) | (np.diff(clock[idx]) > quiet)]
        starts, ends = idx[brk], idx[np.r_[brk[1:], True]]
        for a, b in zip(starts, ends):
            la = max(a - m + 1, int(S["ep_start"][a]))
            f = flag[la:b + 1]
            pp, cc = pred[la:b + 1][f], conf[la:b + 1][f]
            typ = int(np.bincount(pp, minlength=C).argmax())
            rows.append((int(a), int(b), int(la + np.argmax(f)), float(clock[a]), float(clock[b] - clock[a]),
                         int(ep[a]), typ, float(cc.max()), int(f.sum()), bool(truth[la:b + 1].any())))
    al = pd.DataFrame(rows, columns=["start_idx", "end_idx", "first_flag_idx", "raised_t", "duration_s", "episode",
                                     "type_idx", "max_conf", "flagged", "true"])
    al.insert(0, "id", [f"A{seed}-{j + 1:05d}" for j in range(len(al))])
    names, dos, rare = S["names"], set(S["dos_idx"]), S["rare"]
    al["type"] = [names[t] for t in al["type_idx"]]
    al["severity"] = ["low" if cf < LOW_CONF else "critical" if t == rare else "high" if t in dos else "medium"
                      for t, cf in zip(al["type_idx"], al["max_conf"])]
    b = S["bursts"].copy()
    if not b.empty:
        tru = al[al["true"]]
        first = tru.groupby("episode").agg(first_t=("raised_t", "min"), first_id=("id", "first"))
        b = b.join(first, on="episode")
        b["detected"] = b["first_t"].notna()
        b["delay_s"] = np.where(b["detected"], np.maximum(0.0, b["first_t"] - b["start_t"]), np.nan)
        b["class"] = [names[c] if c >= 0 else "-" for c in b["cls"]]
    return al, b


def alert_kpis(S, P, al, b, upto=None):
    span = float(S["clock"][-1] if upto is None else S["clock"][max(0, min(upto, S["n"]) - 1)])
    hours = max(span / 3600.0, 1e-9)
    if upto is not None:
        al = al[al["start_idx"] < upto]
        b = b[b["start_idx"] < upto] if not b.empty else b
    det = float(b["detected"].mean()) if len(b) else np.nan
    return {"detection": det, "fa_h": (~al["true"]).sum() / hours, "alerts_h": len(al) / hours,
            "delay": float(np.nanmedian(b["delay_s"])) if len(b) and b["detected"].any() else np.nan,
            "pps": P["pps"], "hours": span / 3600.0, "n": len(al), "n_false": int((~al["true"]).sum()),
            "bursts": len(b)}


def kpi_bar(K):
    cols = st.columns(6)
    cols[0].metric("Detection rate (bursts)", _pct(K["detection"]),
                   help="Share of real attack bursts with at least one true alert.")
    cols[1].metric("False alarms / hour", _num(K["fa_h"], "{:.1f}"), help="Alerts with no real attack in them, per hour "
                   "of monitored traffic (stream clock).")
    cols[2].metric("Alerts / hour", _num(K["alerts_h"], "{:.1f}"))
    cols[3].metric("Median detection delay", "-" if not np.isfinite(K["delay"]) else f"{K['delay']:.3f} s",
                   help="First true alert time minus the burst's first attack packet time (seconds).")
    cols[4].metric("Packets / s processed", f"{K['pps'] / 1e6:.2f} M", help="Measured model inference speed (CPU).")
    cols[5].metric("Monitored time", f"{K['hours'] * 60:.1f} min" if K["hours"] < 2 else f"{K['hours']:.1f} h")


INCIDENT_WINDOWS = {"60 s": 60.0, "5 min": 300.0, "Off": None}


def group_incidents(al, window):
    """Repeated alerts of one attack type closer than `window` seconds form one incident (alert suppression /
    aggregation as in Suricata `threshold ... seconds` and Elastic Security alert suppression)."""
    if al.empty or window is None:
        return pd.DataFrame()
    a = al.sort_values("raised_t").reset_index(drop=True)
    end = a["raised_t"] + a["duration_s"]
    new = (a["type"] != a["type"].shift()) | (a["raised_t"] - end.shift().cummax() > window)
    a["incident"] = new.cumsum()
    g = a.groupby("incident")
    inc = pd.DataFrame({"first_t": g["raised_t"].min(), "last_t": (a["raised_t"] + a["duration_s"]).groupby(a["incident"]).max(),
                        "type": g["type"].first(), "severity": g["severity"].agg(lambda s: min(s, key=SEV_ORDER.index)),
                        "alerts": g.size(), "true_alerts": g["true"].sum(), "max_conf": g["max_conf"].max(),
                        "first_alert": g["id"].first()})
    inc["true"] = inc["true_alerts"] > 0
    return inc.reset_index(drop=True)


def incidents_panel(al, K, seed, method):
    c = st.columns([1, 1, 1, 1])
    win_l = c[0].selectbox("Group repeats", list(INCIDENT_WINDOWS), 0, key="ids_al_inc",
                           help="Alerts of the same attack type closer than this are one incident (alert suppression).")
    inc = group_incidents(al, INCIDENT_WINDOWS[win_l])
    hours = max(K["hours"], 1e-9)
    if not inc.empty:
        c[1].metric("Incidents / hour", f"{len(inc) / hours:.1f}")
        c[2].metric("False incidents / hour", f"{(~inc['true']).sum() / hours:.1f}")
        c[3].metric("Alerts per incident", f"{inc['alerts'].mean():.1f}")
    if not al.empty:  # alerts over time, pre-binned (one bar per time bin and severity)
        nb = 48
        edges = np.linspace(0.0, max(float(al["raised_t"].max()) / 3600.0, 1e-6), nb + 1)
        rows = []
        for sev in SEV_ORDER:
            h, _ = np.histogram(al.loc[al["severity"] == sev, "raised_t"] / 3600.0, bins=edges)
            rows += [{"t0": edges[i], "t1": edges[i + 1], "severity": sev, "alerts": int(h[i])} for i in range(nb) if h[i]]
        if rows:
            st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                x=alt.X("t0:Q", title="stream time (h)"), x2="t1:Q", y=alt.Y("sum(alerts):Q", title="alerts", stack=True),
                color=alt.Color("severity:N", scale=alt.Scale(domain=SEV_ORDER, range=[SEV_COLORS[x] for x in SEV_ORDER]),
                                legend=alt.Legend(orient="right", title=None)),
                tooltip=["severity", alt.Tooltip("t0:Q", format=".2f", title="from h"), "alerts"]).properties(height=140),
                width="stretch")
    if not inc.empty:
        with st.expander(f"Incidents ({len(inc):,}; {int((~inc['true']).sum()):,} false)"):
            it = pd.DataFrame({"First alert": inc["first_alert"], "Start": [fmt_clock(x) for x in inc["first_t"]],
                               "End": [fmt_clock(x) for x in inc["last_t"]], "Attack": inc["type"],
                               "Severity": inc["severity"], "Alerts": inc["alerts"],
                               "Ground truth": np.where(inc["true"], "true incident", "false incident")})
            st.dataframe(it, hide_index=True, width="stretch", height=300)
    if not al.empty:
        out = al.drop(columns=["type_idx"]).assign(raised_clock=[fmt_clock(x) for x in al["raised_t"]], seed=seed,
                                                   method=method)
        d1, d2, _ = st.columns([1, 1, 4])
        d1.download_button("Export CSV", out.to_csv(index=False), f"alerts_{method}_seed{seed}.csv",
                           "text/csv", key="ids_al_csv")
        d2.download_button("Export JSON", out.to_json(orient="records", lines=True),
                           f"alerts_{method}_seed{seed}.jsonl", "application/json", key="ids_al_json")


def _alert_settings():
    k = st.session_state.get("ids_al_k", 3)
    m = max(st.session_state.get("ids_al_m", 5), k)
    return k, m, float(st.session_state.get("ids_al_q", 1.0))


def tab_alerts(run_dir, meta, methods, seeds):
    run = os.path.basename(run_dir)
    note("Alert console: the global model's packet-level verdicts on the test stream, grouped into alerts the way "
               "an IDS raises them, with severity, ground-truth verdict and analyst triage.")
    have = model_methods(run, methods, seeds)
    if not have:
        st.info("No saved global models in this run yet.")
        return
    dm = default_method(run, tuple(have), tuple(seeds))
    c = st.columns([3, 1, 1, 1, 1])
    method = c[0].selectbox("Method", have, index=have.index(dm) if dm in have else 0, format_func=lab, key="ids_al_mth")
    seed = c[1].selectbox("Seed", list(seeds), key="ids_al_seed")
    k = c[2].slider("k flagged", 1, 10, 3, key="ids_al_k", help="Alert after k of the last m packets are flagged.")
    m = c[3].slider("of last m", 1, 20, 5, key="ids_al_m")
    q = c[4].number_input("Quiet gap (s)", 0.0, 60.0, 1.0, 0.5, key="ids_al_q",
                          help="An alert closes when no confirmed packet follows for this long.")
    if m < k:
        st.caption(f"m < k: using m = k = {k}.")
        m = k
    purpose(f"confirmation rule (standard IDS practice): an alert opens when at least {k} of the last {m} packets of "
            f"one link timeline are flagged as an attack, and closes after {q:g} s without confirmed packets.")
    S = _stream_or_warn(run, seed)
    if S is None:
        return
    P = predictions(run, seed, method)
    al, b = build_alerts(run, seed, method, k, m, float(q))
    live = st.session_state.get("ids_live") or {}
    upto = None
    if live.get("key") == (run, seed) and st.toggle("Only up to the Live replay position", False, key="ids_al_upto",
                                                   help="Show what the console would have shown so far in the Live tab."):
        upto = int(live.get("pos", 0))
    K = alert_kpis(S, P, al, b, upto)
    kpi_bar(K)
    purpose("detection rate = real attack bursts that got at least one true alert; false alarms / hour and alerts / "
            f"hour use the packets' timestamps (idle gaps longer than {IDLE_CAP:g} s are shortened to {IDLE_CAP:g} s on "
            "the stream clock); delay is measured in seconds from the burst's first attack packet.")
    if not S["has_time"]:
        st.caption("This run's test set has no per-packet timeline: times come from the gap feature, one stream.")
    incidents_panel(al if upto is None else al[al["start_idx"] < upto], K, seed, method)

    view = al if upto is None else al[al["start_idx"] < upto]
    triage = st.session_state.setdefault("ids_triage", {})
    f1, f2, f3, f4 = st.columns([3, 2, 2, 1])
    sev_f = f1.multiselect("Severity", SEV_ORDER, SEV_ORDER, key="ids_al_sev")
    ver_f = f2.selectbox("Ground truth", ["all", "true alerts", "false alarms"], key="ids_al_ver")
    st_f = f3.selectbox("Analyst status", ["all", "open", "acknowledged", "false positive"], key="ids_al_st")
    limit = f4.selectbox("Rows", [100, 500, 2000], 1, key="ids_al_lim")
    tkey = f"{run}|{seed}|{method}|{k}/{m}/{q:g}"  # alert ids depend on the confirmation settings
    view = view.assign(status=[triage.get(f"{tkey}|{a}", "open") for a in view["id"]])
    view = view[view["severity"].isin(sev_f)]
    if ver_f != "all":
        view = view[view["true"] == (ver_f == "true alerts")]
    if st_f != "all":
        view = view[view["status"] == st_f]
    view = view.sort_values("raised_t", ascending=False).head(limit)
    sev_counts = al["severity"].value_counts()
    st.markdown(" ".join(f"{sev_badge(s)} <span class='ids-kv'>{int(sev_counts.get(s, 0)):,}</span>&nbsp;&nbsp;"
                         for s in SEV_ORDER) + f"<span class='ids-kv'> · {len(al):,} alerts, "
                f"{int((~al['true']).sum()):,} false · showing {len(view):,} (newest first)</span>",
                unsafe_allow_html=True)
    table = pd.DataFrame({"Alert": view["id"], "Raised (stream clock)": [fmt_clock(t) for t in view["raised_t"]],
                          "Duration (s)": view["duration_s"].round(3), "Source": f"global model · seed {seed}",
                          "Predicted attack": view["type"], "Severity": view["severity"],
                          "Max conf.": view["max_conf"].round(3), "Packets flagged": view["flagged"],
                          "Ground truth": np.where(view["true"], "true alert", "false alarm"), "Status": view["status"]})

    def sev_style(v):
        return f"background-color:{SEV_COLORS[v]};color:{SEV_TEXT[v]};font-weight:700" if v in SEV_COLORS else ""

    def gt_style(v):
        return "color:#d64545;font-weight:600" if v == "false alarm" else ""
    ev = st.dataframe(table.style.map(sev_style, subset=["Severity"]).map(gt_style, subset=["Ground truth"])
                      .format({"Duration (s)": "{:.3f}", "Max conf.": "{:.3f}"}),
                      hide_index=True, width="stretch", height=min(420, 35 * (len(table) + 1) + 3),
                      on_select="rerun", selection_mode="multi-row", key=f"ids_al_tbl_{tkey}")
    sel = []
    try:
        sel = [table["Alert"].iloc[i] for i in ev.selection.rows]
    except Exception:
        pass
    b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
    for col, label_, new in ((b1, "Acknowledge", "acknowledged"), (b2, "Mark false positive", "false positive"),
                             (b3, "Reopen", "open")):
        if col.button(label_, key=f"ids_al_{new}", disabled=not sel, width="stretch"):
            for a in sel:
                triage[f"{tkey}|{a}"] = new
            st.rerun()
    mine = {kk.split("|")[-1]: v for kk, v in triage.items() if kk.startswith(tkey + "|")}
    if mine:
        fp = [a for a, v in mine.items() if v == "false positive"]
        truth_map = dict(zip(al["id"], al["true"]))
        n_fa = sum(not truth_map.get(a, True) for a in fp)
        b4.caption(f"Triaged: {sum(v == 'acknowledged' for v in mine.values())} acknowledged, {len(fp)} marked false "
                   f"positive ({n_fa} of them are false alarms by ground truth).")
    purpose("select rows, then acknowledge or mark them false positive (kept for this browser session only; nothing is "
            "written to disk). Severity: rare DoS type = critical, other DoS types = high, non-DoS 'Other attack' = "
            f"medium, any alert whose strongest packet is below {LOW_CONF:.0%} confidence = low. Ground truth = the alert "
            "overlaps a real attack burst. Source: the test stream is one shared link stream scored by the global model "
            "each drone runs; it is not split by drone.")
    if not b.empty:
        with st.expander(f"Attack bursts in this test stream ({len(b)}): detected? delay?"):
            bt = pd.DataFrame({"Burst (episode)": b["episode"], "Class": b["class"],
                               "Start": [fmt_clock(t) for t in b["start_t"]], "Attack packets span": b["packets"],
                               "Detected": np.where(b["detected"], "yes", "NO"), "Delay (s)": b["delay_s"],
                               "First alert": b["first_id"].fillna("-")})
            per_cls = b.groupby("class").agg(bursts=("detected", "size"), detected=("detected", "mean"),
                                             median_delay_s=("delay_s", "median")).reset_index()
            st.dataframe(per_cls.style.format({"detected": "{:.0%}", "median_delay_s": "{:.3f}"}), hide_index=True)
            st.dataframe(bt.style.format({"Delay (s)": "{:.3f}"}), hide_index=True, width="stretch", height=300)
            purpose("each real attack burst inserted into normal traffic in the test stream: whether any true alert was "
                    "raised and how long after its first attack packet.")


# ---------- Live tab ----------
SPEEDS = {"10 packets/s": 10, "100 packets/s": 100, "1,000 packets/s": 1000, "10,000 packets/s": 10_000,
          "100,000 packets/s": 100_000, "Real time": None}


def _live_state(run, seed):
    s = st.session_state.get("ids_live")
    if not s or s.get("key") != (run, seed):
        s = {"key": (run, seed), "pos": 0, "segs": [[0, 0]], "playing": False, "tick": time.time(), "acc": 0.0,
             "note": ""}
        st.session_state["ids_live"] = s
    return s


def _move(s, new_pos, n, new_segment=False):
    new_pos = int(min(max(new_pos, 0), n))
    if new_segment:
        s["segs"].append([new_pos, new_pos])
    else:
        seg = s["segs"][-1]
        if new_pos < seg[0]:
            seg[0] = new_pos
        seg[1] = new_pos
    s["pos"] = new_pos


def _seen_idx(s):
    return [(a, b) for a, b in s["segs"] if b > a]


def _running(S, pred, segs):
    C, nt = S["C"], S["normal"]
    cm = np.zeros((C, C), dtype=np.int64)
    for a, b in segs:
        cm += np.bincount(S["y"][a:b] * C + pred[a:b], minlength=C * C).reshape(C, C)
    return cm


def _result(t, p, nt):
    if t == nt:
        return ("ok", "") if p == nt else ("FALSE ALARM", "fa")
    if p == nt:
        return ("MISSED", "miss")
    return ("DETECTED", "tp") if p == t else ("WRONG TYPE", "wt")


def _so_far(raised, S, P, pos, m):
    """Alerts as known at the replay position: an alert still open is summarised only over the packets seen so far."""
    if not len(raised):
        return raised
    raised = raised.copy()
    nt, names, dos, rare = S["normal"], S["names"], set(S["dos_idx"]), S["rare"]
    truth = (S["y"] != nt) | S["att"]
    for j, r in raised[raised["end_idx"] >= pos].iterrows():
        la = max(int(r["start_idx"]) - m + 1, int(S["ep_start"][int(r["start_idx"])]))
        f = P["pred"][la:pos] != nt
        if not f.any():
            continue
        pp, cc = P["pred"][la:pos][f], P["conf"][la:pos][f]
        t = int(np.bincount(pp, minlength=S["C"]).argmax())
        raised.loc[j, ["flagged", "max_conf", "type", "true", "duration_s"]] = [
            int(f.sum()), float(cc.max()), names[t], bool(truth[la:pos].any()),
            float(S["clock"][pos - 1] - S["clock"][int(r["start_idx"])])]
        raised.loc[j, "severity"] = ("low" if cc.max() < LOW_CONF else "critical" if t == rare else
                                     "high" if t in dos else "medium")
    return raised


def tab_live(run_dir, meta, methods, seeds):
    run = os.path.basename(run_dir)
    note("Live replay: the test stream played back packet by packet as a sensor on the drone link would see it. "
               "Predictions are computed once per method and only revealed as the replay advances.")
    have = model_methods(run, methods, seeds)
    if not have:
        st.info("No saved global models in this run yet.")
        return
    dm = default_method(run, tuple(have), tuple(seeds))
    c = st.columns([3, 1, 2, 2])
    method = c[0].selectbox("Method", have, index=have.index(dm) if dm in have else 0, format_func=lab, key="ids_lv_m")
    seed = c[1].selectbox("Seed", list(seeds), key="ids_lv_s")
    speed_l = c[2].selectbox("Speed", list(SPEEDS), 1, key="ids_lv_speed",
                             help=f"Real time follows the packets' own timestamps (waits longer than "
                                  f"{REPLAY_SLEEP_CAP:g} s are shortened, like tcpreplay --maxsleep).")
    compare = c[3].toggle("Compare methods", False, key="ids_lv_cmp",
                          help="The same packets through FedAvg, Multi-Krum and the selected method side by side.")
    S = _stream_or_warn(run, seed)
    if S is None:
        return
    s = _live_state(run, seed)
    n = S["n"]
    b = st.columns([1.2, 1, 1, 1, 1, 1, 2.2, 1.6])
    shown = "Pause" if s["playing"] else "Play"
    if b[0].button(shown, type="primary", key="ids_lv_play", width="stretch"):
        s["playing"] = not s["playing"]
        s["tick"], s["acc"] = time.time(), 0.0
    if b[1].button("Reset", key="ids_lv_reset", width="stretch"):
        s.update(pos=0, segs=[[0, 0]], playing=False, note="")
    for col, d in zip(b[2:6], (-100, -1, 1, 100)):
        if col.button(f"{d:+d}", key=f"ids_lv_step{d}", width="stretch", help=f"Step {d:+d} packet(s)"):
            s["playing"] = False
            _move(s, s["pos"] + d, n)
    if shown != ("Pause" if s["playing"] else "Play"):
        st.rerun()  # redraw the button with its new label (Play <-> Pause)
    bursts = S["bursts"]
    att_classes = sorted({int(x) for x in bursts["cls"] if x >= 0}) if not bursts.empty else []
    inj_cls = b[6].selectbox("Attack class", att_classes, format_func=lambda i: S["names"][i], key="ids_lv_inj",
                             label_visibility="collapsed") if att_classes else None
    if b[7].button("Inject burst", key="ids_lv_jump", width="stretch", disabled=inj_cls is None,
                   help="Jump the replay to the next real test burst of this attack class."):
        cand = bursts[bursts["cls"] == inj_cls]
        nxt = cand[cand["start_idx"] > s["pos"]]
        row = (nxt if len(nxt) else cand).iloc[0]
        b_idx, b_ep, b_pk = int(row["start_idx"]), int(row["episode"]), int(row["packets"])
        start = max(int(S["ep_start"][b_idx]), b_idx - 20)
        _move(s, start, n, new_segment=True)
        s["note"] = (f"Injected: real {S['names'][inj_cls]} burst (episode {b_ep}, {b_pk:,} packets) "
                     f"starting at {fmt_clock(float(row['start_t']))}; replay placed {b_idx - start} packets before it.")
    purpose("Play / Pause, step back or forward, or inject an attack burst (jumps to a real burst of that class in "
            "the test stream; nothing is synthesised). Running numbers count only the packets replayed so far.")
    if compare:
        cmp_m = [m for m in ("fedavg", "multi_krum") if m in have and m != method] + [method]
    else:
        cmp_m = [method]
    try:
        preds = {m: predictions(run, seed, m) for m in cmp_m}
    except Unavailable as e:
        st.warning(str(e))
        return
    st.fragment(run_every=1.0 if s["playing"] else None)(_live_panel)(run, seed, method, cmp_m, preds, SPEEDS[speed_l])


def _live_panel(run, seed, method, cmp_m, preds, speed):
    S = load_stream(run, seed)
    s = _live_state(run, seed)
    n, nt, names = S["n"], S["normal"], S["names"]
    if s["playing"]:
        now = time.time()
        dt = min(max(now - s["tick"], 0.0), 2.0)
        s["tick"] = now
        if speed is None:
            target = (S["rclock"][s["pos"] - 1] if s["pos"] else 0.0) + dt
            new = int(np.searchsorted(S["rclock"], target, side="right"))
        else:
            s["acc"] += speed * dt
            adv = int(s["acc"])
            s["acc"] -= adv
            new = s["pos"] + adv
        _move(s, new, n)
        if s["pos"] >= n:
            s["playing"] = False
    pos = s["pos"]
    segs = _seen_idx(s)
    seen = sum(b - a for a, b in segs)
    P = preds[method]
    k, m, q = _alert_settings()
    al, _ = build_alerts(run, seed, method, k, m, q)
    raised = al[(al["start_idx"] < pos) & al["start_idx"].apply(lambda i: any(a <= i < b for a, b in segs))] \
        if len(al) and segs else al.iloc[0:0]
    open_now = raised["end_idx"] >= pos - 1 if len(raised) else None
    raised = _so_far(raised, S, P, pos, m)
    now_t = float(S["clock"][pos - 1]) if pos else 0.0
    state = "PLAYING" if s["playing"] else ("END" if pos >= n else "PAUSED")
    st.markdown(f'<div class="ids-kv">{state} · packet {pos:,} / {n:,} · stream clock {fmt_clock(now_t)} · '
                f'{seen:,} packets replayed · alert rule {k} of last {m}, quiet {q:g} s</div>', unsafe_allow_html=True)
    if s.get("note"):
        st.caption(s["note"])
    # Alert banner: the most recent alert of the primary method, if it is still open at the replay position.
    active = raised[open_now.to_numpy() | (S["clock"][np.minimum(raised["end_idx"].to_numpy(dtype=int), max(pos - 1, 0))]
                                         >= now_t - q)] if len(raised) else raised
    if len(active):
        a = active.iloc[-1]
        col = SEV_COLORS[a["severity"]]
        st.markdown(f'<div class="ids-banner" style="border-left-color:{col}">{sev_badge(a["severity"])} '
                    f'<b>ALERT {esc(a["id"])}</b> · {esc(a["type"])} detected by {esc(lab(method))} · raised '
                    f'{fmt_clock(a["raised_t"])} · max confidence {a["max_conf"]:.2f} · {int(a["flagged"])} packets '
                    f'flagged so far · ground truth: {"real attack" if a["true"] else "FALSE ALARM"}</div>',
                    unsafe_allow_html=True)
    else:
        last = f" · last alert {raised.iloc[-1]['id']} at {fmt_clock(raised.iloc[-1]['raised_t'])}" if len(raised) else ""
        st.markdown(f'<div class="ids-banner" style="border-left-color:rgba(128,128,128,.6)">no active alert · '
                    f'monitoring{esc(last)}</div>', unsafe_allow_html=True)

    st.markdown("##### Running KPIs")
    krows = []
    for mm in cmp_m:
        cmx = _running(S, preds[mm]["pred"], segs)
        M = metrics_from_cm(S, cmx)
        alm = raised if mm == method else None
        if alm is None:
            a2, _ = build_alerts(run, seed, mm, k, m, q)
            alm = a2[(a2["start_idx"] < pos) & a2["start_idx"].apply(lambda i: any(a <= i < b for a, b in segs))] \
                if len(a2) and segs else a2.iloc[0:0]
            alm = _so_far(alm, S, preds[mm], pos, m)
        krows.append({"Method": lab(mm), "Accuracy": _pct(M["accuracy"]),
                      "Attacks detected": _pct(1 - M["missed"]) if np.isfinite(M["missed"]) else "-",
                      "Missed": _pct(M["missed"]), "False alarms": _pct(M["false_alarms"]),
                      "Alerts": len(alm), "False alerts": int((~alm["true"]).sum()) if len(alm) else 0})
    kt = pd.DataFrame(krows)
    head = "".join(f"<th>{esc(c)}</th>" for c in kt.columns)
    rows = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>" for r in kt.itertuples(index=False))
    st.markdown(f"<div class='ids-scroll'><table class='ids'><tr>{head}</tr>{rows}</table></div>", unsafe_allow_html=True)
    purpose(f"over the {seen:,} packets replayed so far: packet accuracy, attack packets flagged / called Normal, "
            "normal packets flagged, and alerts raised (k-of-m rule from the Alerts tab).")
    st.markdown("##### Packet log (newest first)")
    with st.container():
        lo = max(0, pos - 18)
        rows_i = list(range(pos - 1, lo - 1, -1))
        size = S["size"]
        hdr = "<th>time</th><th class='r'>#</th><th class='r'>size B</th><th class='r'>gap ms</th><th>true</th>"
        if len(cmp_m) == 1:
            hdr += "<th>predicted</th><th class='r'>conf</th><th>result</th><th>match</th>"
        else:
            hdr += "".join(f"<th>{esc(lab(mm)[:22])}</th>" for mm in cmp_m)
        body = []
        for i in rows_i:
            t = int(S["y"][i])
            cells = (f"<td>{fmt_clock(S['clock'][i])}</td><td class='r'>{i:,}</td>"
                     f"<td class='r'>{'-' if size is None else f'{size[i]:,.0f}'}</td>"
                     f"<td class='r'>{S['gap'][i] * 1000:,.3f}</td><td>{esc(names[t])}</td>")
            if len(cmp_m) == 1:
                p, cf = int(P["pred"][i]), float(P["conf"][i])
                res, cls = _result(t, p, nt)
                cells += (f"<td>{esc(names[p])}</td><td class='r'>{cf:.2f}</td><td>{res}</td>"
                          f"<td>{'yes' if p == t else 'NO'}</td>")
                body.append(f"<tr class='{cls}'>{cells}</tr>")
            else:
                for mm in cmp_m:
                    p, cf = int(preds[mm]["pred"][i]), float(preds[mm]["conf"][i])
                    res, cls = _result(t, p, nt)
                    cells += f"<td class='{cls}'>{esc(names[p][:14])} {cf:.2f}{'' if res == 'ok' else ' · ' + res}</td>"
                body.append(f"<tr>{cells}</tr>")
        if not body:
            body = [f"<tr><td colspan='9' style='opacity:.6'>press Play or +1 to start the replay</td></tr>"]
        st.markdown(f"<div class='ids-scroll'><table class='ids'><tr>{hdr}</tr>{''.join(body)}</table></div>", unsafe_allow_html=True)
        purpose("each test packet as it arrives: stream time, size and gap since the previous packet (from the "
                "capture), true label, the model's prediction and confidence. Shading: orange = attack detected, dark "
                "red = attack missed (called Normal), yellow = false alarm on normal traffic.")
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"##### Running confusion matrix · {esc(lab(method))}", unsafe_allow_html=True)
        st.markdown(html_cm(_running(S, P["pred"], segs), names), unsafe_allow_html=True)
        purpose("counts so far (row share in grey); blue diagonal = correct, red = confusions.")
    with right:
        if not len(raised):
            st.markdown("##### Alerts raised so far")
            st.caption("none yet")
        if len(raised):
            st.markdown("##### Alerts raised so far (latest 8)")
            last = raised.tail(8).iloc[::-1]
            rows = "".join(f"<tr><td>{esc(r.id)}</td><td>{fmt_clock(r.raised_t)}</td><td class='r'>{r.duration_s:.3f}</td>"
                           f"<td>{sev_badge(r.severity)}</td><td>{esc(r.type)}</td><td class='r'>{r.max_conf:.2f}</td>"
                           f"<td class='r'>{r.flagged}</td><td>{'true alert' if r.true else 'FALSE ALARM'}</td></tr>"
                           for r in last.itertuples())
            st.markdown("<div class='ids-scroll'><table class='ids'><tr><th>alert</th><th>raised</th><th class='r'>dur s</th><th>severity</th>"
                        "<th>predicted</th><th class='r'>conf</th><th class='r'>pkts</th><th>ground truth</th></tr>"
                        f"{rows}</table></div>", unsafe_allow_html=True)
            purpose("the newest alerts of the selected method up to the replay position; the full list with triage is in "
                    "the Alerts tab.")


# ---------- Swarm monitor: why dropped / readmitted ----------
def why_panel(meta, method, seed, r, rounds, idx):
    with st.expander("Why was this drone dropped / readmitted?"):
        purpose("explains one drone's decision in this round from the logged fields only: the filter score and rank, "
                "the vote count against the majority needed, and the DQN's inputs and chosen weight.")
        clients = r.get("clients", [])
        if not clients:
            st.info("No per-drone record in this round.")
            return
        cfg = meta.get("config", {}) or {}
        N = len(clients)
        role_txt = {"general": "regular", "rare": f"rare ({meta.get('rare_class', 'rare class')})", "poisoned": "poisoned"}
        cid = st.selectbox("Drone", [d["cid"] for d in clients], key="ids_why_d",
                           format_func=lambda c: f"Drone {c + 1:02d} · {role_txt.get(clients[c]['role'], '?')}"
                           if c < len(clients) else f"Drone {c + 1:02d}")
        d = next(x for x in clients if x["cid"] == cid)
        scores = {x["cid"]: x.get("score") for x in clients if x.get("score") is not None}
        facts = [("round", f"{r['round'] + 1}"), ("status this round", d.get("status", "-")),
                 ("share of the global model", f"{d.get('weight', 0):.1%}")]
        rule, K, kept_ids, chart_rule = "", None, None, None
        if method.startswith("jiang") or method == "multi_krum":
            K = (cfg.get("jiang", {}) if method.startswith("jiang") else cfg.get("multi_krum", {})).get("K", 5)
            order = sorted(scores, key=lambda c: scores[c])
            kept_ids = set(order[:K])
            rule = ("Jiang filter: total squared distance of the model to all other models; the K most central "
                    "(smallest) are kept" if method.startswith("jiang") else
                    "Multi-Krum: sum of squared distances to the N-f-2 nearest models; the K smallest are kept")
            if cid in scores:
                rank = order.index(cid) + 1
                facts += [("filter score", f"{scores[cid]:.3f}"), ("rank (1 = most central)", f"{rank} of {len(order)}"),
                          ("kept by filter (rank <= K)", f"{'yes' if rank <= K else 'no'} (K = {K})")]
                if len(order) > K:
                    chart_rule = (scores[order[K - 1]] + scores[order[K]]) / 2
        elif method == "fedwgca":
            fc = cfg.get("fedwgca", {})
            qq, lam = fc.get("quartile", 0.25), fc.get("iqr_lambda", 1.5)
            v = np.array(list(scores.values()))
            if len(v):
                q1, q3 = np.percentile(v, [100 * qq, 100 * (1 - qq)])
                lo_, hi_ = q1 - lam * (q3 - q1), q3 + lam * (q3 - q1)
                kept_ids = {c for c, s_ in scores.items() if lo_ <= s_ <= hi_}
                rule = f"FedWGCA: update size (norm) must lie inside the IQR fences [{lo_:.3f}, {hi_:.3f}]"
                if cid in scores:
                    facts += [("update norm", f"{scores[cid]:.3f}"),
                              ("inside fences", "yes" if cid in kept_ids else "no")]
        else:
            rule = "This method has no filter (every model is averaged), so there is no filter score."
        st.caption(rule)
        if "votes_total" in d:
            need = N // 2 + 1
            facts += [("vote on model of round", f"{d.get('vote_round', r['round']) + 1}"),
                      ("votes yes / total", f"{d['votes_yes']} / {d['votes_total']}"),
                      ("majority needed (> N/2)", f">= {need} of {N}"),
                      ("readmitted", "yes" if d.get("readmitted") else "no")]
        elif method.startswith("jiang"):
            facts.append(("vote", "no vote on this drone this round (kept by the filter, or voting not started)"))
        fixed_w = (cfg.get("jiang", {}) or {}).get("fixed_weight") or 1.0 / ((cfg.get("jiang", {}) or {}).get("K", 5) + 1)
        if d.get("dqn_weight") is not None:
            mult = d["dqn_weight"] / fixed_w if fixed_w else np.nan
            facts += [("Eq. 9 weight given", f"{d['dqn_weight']:.3f} (= {mult:.2g} x fixed 1/(K+1))")]
        if d.get("dqn_state"):
            stt = list(d["dqn_state"]) + [None] * 4
            facts += [("DQN state: training progress", _num(stt[0])), ("DQN state: model age (r'/r)", _num(stt[1])),
                      ("DQN state: public-exam score", _num(stt[2])), ("DQN state: yes-vote share", _num(stt[3])),
                      ("DQN reward", _num(d.get("reward"), "{:+.4f}"))]
            acts = ((cfg.get("jiang", {}) or {}).get("dqn", {}) or {}).get("actions")
            if acts:
                facts.append(("DQN action set (x fixed weight)", ", ".join(f"{a:g}" for a in acts)))
        c1, c2 = st.columns([1, 1])
        with c1:
            rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in facts)
            st.markdown(f"<div class='ids-scroll'><table class='ids'><tr><th>field</th><th>value</th></tr>{rows}</table></div>",
                        unsafe_allow_html=True)
        with c2:
            if scores:
                df = pd.DataFrame([{"Drone": f"D{c + 1:02d}", "Score": s_, "Role": role_txt.get(clients[c]["role"], "?")
                                    if c < len(clients) else "?",
                                    "Kept": ("this drone" if c == cid else "kept" if kept_ids and c in kept_ids else "dropped")}
                                   for c, s_ in scores.items()])
                ch = alt.Chart(df).mark_bar().encode(
                    x=alt.X("Drone:N", sort=alt.EncodingSortField("Score", order="ascending"), title=None,
                            axis=alt.Axis(labelColor=app_roles.label_color({f"D{c + 1:02d}": clients[c]["role"]
                                                                            for c in scores if c < len(clients)}))),
                    y=alt.Y("Score:Q", title="filter score (lower = more typical)" if method != "fedwgca" else "update norm"),
                    color=alt.Color("Kept:N", scale=alt.Scale(domain=["this drone", "kept", "dropped"],
                                                              range=["#1565c0", "#78909c", "#c62828"]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Drone", "Role", alt.Tooltip("Score:Q", format=".3f"), "Kept"])
                if chart_rule is not None:
                    ch += alt.Chart(pd.DataFrame({"y": [chart_rule]})).mark_rule(strokeDash=[4, 3], color="gray") \
                        .encode(y="y:Q")
                st.altair_chart(ch.properties(height=220), width="stretch")
                purpose("every drone's score this round, sorted; dashed line = the cut between the K kept and the rest.")
        hist = [x for rr in rounds[max(0, idx - 49):idx + 1] for x in rr.get("clients", []) if x["cid"] == cid]
        if hist:
            cnt = pd.Series([x.get("status", "?") for x in hist]).value_counts()
            st.caption(f"Last {len(hist)} rounds for this drone: " + ", ".join(f"{k} {v}" for k, v in cnt.items()) + ".")
        note("Per-voter scores and individual yes/no votes were not logged (only the yes count, the total and "
                   "the outcome), so which drones voted for this model cannot be shown.")


# ---------- Verification tab ----------
CHECKS = [
    ("Unit tests", lambda run: [PY, "-m", "pytest", "-q", "tests"],
     "pytest on tests/: aggregation rules, voting, data splits and metrics behave as specified on small known inputs."),
    ("Goal checks (test set)", lambda run: [PY, "check_goals.py", run, "--test"],
     "PASS / FAIL per project goal on the test set: per seed, per drone and per class (poison kept out, rare drone "
     "kept, no collapsed class, no suspiciously perfect matrix)."),
    ("Expected trends", lambda run: [PY, "tools/trend_check.py", run],
     "OK / ISSUE per expected trend between methods (Jiang vs FedAvg / Multi-Krum / FedWGCA / Fed-GraB, DQN order)."),
]


def _colour(out):
    lines = []
    for line in out.splitlines():
        e = esc(line)
        s = line.strip()
        if s.startswith(("PASS", "OK")) or " passed" in s and "failed" not in s:
            lines.append(f'<span class="ok">{e}</span>')
        elif s.startswith(("FAIL", "ISSUE", "ERROR", "E ")) or " failed" in s:
            lines.append(f'<span class="bad">{e}</span>')
        else:
            lines.append(f'<span class="dim">{e}</span>' if not s else e)
    return "\n".join(lines)


def _run_check(cmd):
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1"}
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900, env=env)
        out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
    except Exception as e:
        out, rc = f"could not run: {e}", -1
    return {"out": out, "rc": rc, "secs": time.perf_counter() - t0, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cmd": " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd)}


def tab_verification(run_dir, meta):
    run = os.path.basename(run_dir)
    note("Testing evidence for the review: run the project's automatic checks on this run now. Read-only: they "
               "only read saved results; nothing is trained or written.")
    store = st.session_state.setdefault("ids_verif", {})
    run_all = st.button("Run all three", type="primary", key="ids_vf_all")
    for i, (name, mk, help_) in enumerate(CHECKS):
        st.markdown(f"#### {name}")
        st.caption(help_)
        cmd = mk(run)
        if st.button(f"Run: {' '.join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd)}",
                     key=f"ids_vf_{i}") or run_all:
            with st.spinner(f"Running {name}..."):
                store[(run, i)] = _run_check(cmd)
        res = store.get((run, i))
        if res:
            out = res["out"]
            n_ok = sum(1 for l in out.splitlines() if l.strip().startswith(("PASS", "OK")))
            n_bad = sum(1 for l in out.splitlines() if l.strip().startswith(("FAIL", "ISSUE")))
            summary = f"exit code {res['rc']} · {res['secs']:.1f} s · {res['when']}"
            if n_ok or n_bad:
                summary += f" · {n_ok} PASS/OK · {n_bad} FAIL/ISSUE"
            st.markdown(f'<div class="ids-kv">$ {esc(res["cmd"])} · {esc(summary)}</div>'
                        f'<div class="ids-out">{_colour(out)}</div>', unsafe_allow_html=True)
    purpose("green lines passed, red lines failed. FAIL / ISSUE lines are real findings about the results (for example a "
            "class that stays hard to detect); they are reported, not hidden.")
