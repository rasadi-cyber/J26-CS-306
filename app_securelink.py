"""Secure Link panel: animated view of the encrypted drone <-> aggregation-server link.

Pure presentation: reads `data/demo/demo_trace.json` (made by tools/make_demo_trace.py from a real short run) and, for
runs trained from the dashboard with `security.secure_link` on, each job's `securelink.jsonl` / `live_state.json` /
`rounds.jsonl`. Never imports the training code. Everything is drawn in one self-contained HTML component (inline SVG,
CSS and JS; no external assets), so playing, pausing and stepping never cause a Streamlit rerun.

CPU: the page is idle until the viewer presses play (a static frame, no animation loop). While playing it draws at
~30 fps and stops drawing when paused or finished, when the browser tab is hidden and when the frame is off-screen or in
a hidden tab (IntersectionObserver), so several of these frames on one page cost nothing until one is played.

Public:
    render_demo(trace_path=DEMO_TRACE, height=780)   full demo (about 2.5 min) + one button per step
    render_secure_link(run_dir, meta, height=720)    live / replay view of a real run's link log
"""
import json
import math
import os

import streamlit as st
import streamlit.components.v1 as components

try:  # shared with the Drone Sites panel: same names and grid (km east / north of the server at 0, 0)
    from app_sites import SITES, SERVER, site, drone_name
except Exception:  # app_sites missing or failing to import: positions from the trace, generic names
    SITES, SERVER, site = [], {"name": "Aggregation server", "x": 0.0, "y": 0.0}, None

    def drone_name(cid):
        return f"Drone {cid + 1:02d}"

DEMO_TRACE = os.path.join("data", "demo", "demo_trace.json")
LINK_LOG, LIVE_STATE, ROUNDS = "securelink.jsonl", "live_state.json", "rounds.jsonl"


# ---------- files ----------
def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _mtime(path):
    try:
        s = os.stat(path)
        return (s.st_mtime_ns, s.st_size)
    except OSError:
        return None


@st.cache_data(show_spinner=False, max_entries=8)
def _read_jsonl(path, stamp):
    """All complete lines of a .jsonl file (a half-written last line while training is skipped)."""
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except OSError:
        pass
    return out


def _jsonl(path):
    stamp = _mtime(path)
    return _read_jsonl(path, stamp) if stamp else []


# ---------- trace -> compact payload for the page ----------
def _f(x, nd=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


def _floats(xs):
    return [_f(x) for x in (xs or [])][:8]


def _cid(x):
    return int(x.get("cid", -1))


def _sites(trace_sites=None, n=None):
    """Names and grid positions (km east / north of the server) from app_sites, so both panels match. Without
    app_sites: the trace's 0..1 map positions (y down) turned into the same kind of grid, else a ring."""
    extra = {int(s["cid"]): s for s in (trace_sites or []) if "cid" in s}
    cids = sorted(extra) or list(range(n or len(SITES) or 10))
    out = []
    for i, c in enumerate(cids):
        e = extra.get(c, {})
        if site is not None:
            b = site(c)
            x, y, name = b["x"], b["y"], b["name"]
        elif e.get("x") is not None and e.get("y") is not None:
            x, y, name = (e["x"] - 0.5) * 10, (0.5 - e["y"]) * 10, e.get("name")
        else:
            a = math.pi / 2 - 2 * math.pi * i / len(cids)
            x, y, name = 4.2 * math.cos(a), 4.2 * math.sin(a), e.get("name")
        out.append({"cid": c, "name": name or f"Site {c + 1}", "label": drone_name(c), "x": float(x), "y": float(y),
                    "role": e.get("role"), "n": e.get("n_train")})
    return out


def _up(u):
    return {"cid": _cid(u), "bytes": u.get("bytes"), "enc": _f(u.get("enc_ms"), 3), "dec": _f(u.get("dec_ms"), 3),
            "att": u.get("attempts") or 1,
            "ok": bool(u.get("verify_ok", not u.get("error"))), "err": u.get("error"), "tx": _f(u.get("transfer_s"), 4),
            "nonce": u.get("nonce_hex") or "", "tag": u.get("tag_hex") or "", "sig": u.get("sig_hex") or "",
            "ph": u.get("plain_hex") or "", "ch": u.get("cipher_hex") or "",
            "ps": _floats(u.get("plain_sample")), "ds": _floats(u.get("decrypted_sample"))}


def _row(x):
    return {"cid": _cid(x), "role": x.get("role"), "status": x.get("status"), "w": _f(x.get("weight"), 6) or 0.0,
            "score": _f(x.get("score"), 4), "yes": x.get("yes", x.get("votes_yes")), "total": x.get("votes_total"),
            "dqn": _f(x.get("dqn_weight"), 4)}


def _round(r, rec=None):
    """One round in the page's compact shape. `rec`: that round's rounds.jsonl row, used when the line (a
    securelink.jsonl line) has no server part."""
    rec = rec or {}
    srv = r.get("server") or {"global_sample": r.get("global_sample"), "val": rec.get("val")}
    rr = [_row(x) for x in (srv.get("rows") or rec.get("clients") or [])]
    sel = srv.get("selected")
    if sel is None:
        sel = [x["cid"] for x in rr if x["status"] == "accepted"]
    drop = srv.get("dropped")
    if drop is None:
        drop = [x["cid"] for x in rr if x["status"] != "accepted"]
    readm = srv.get("readmitted")
    if readm is None:
        readm = [x["cid"] for x in rr if x["status"] == "readmitted"]
    dqn = {str(k): _f(v, 4) for k, v in (srv.get("dqn_weights") or {}).items()}
    if not dqn:
        dqn = {str(x["cid"]): x["dqn"] for x in rr if x["cid"] in readm and x["dqn"] is not None}
    vote = r.get("vote") or {}
    tot = r.get("totals") or {}
    up = [_up(u) for u in (r.get("uplink") or [])]
    local = r.get("local") or [u for u in (r.get("uplink") or []) if "local_train_ms" in u]  # link log: timings in uplink
    down = [{"cid": _cid(d), "bytes": d.get("bytes"), "tx": _f(d.get("transfer_s"), 4), "att": d.get("attempts") or 1}
            for d in (r.get("downlink") or [])]
    events = [{"cid": _cid(e), "dir": e.get("dir"), "attempt": e.get("attempt"), "error": e.get("error"),
               "byte": e.get("byte"), "nack": e.get("nack_bytes"), "resend": e.get("resend_bytes"),
               "ok": bool(e.get("recovered"))} for e in (r.get("events") or [])]
    val = srv.get("val") or {}
    return {"round": int(r.get("round", 0)),
            "local": [{"cid": _cid(x), "ms": _f(x.get("local_train_ms"), 1), "steps": x.get("n_steps"),
                       "norm": _f(x.get("update_norm"), 4), "loss": _f(x.get("loss_hint"), 4)} for x in local],
            "up": up, "down": down, "rows": rr, "sel": [int(c) for c in sel], "drop": [int(c) for c in drop],
            "readm": [int(c) for c in readm], "dqn": dqn, "g": _floats(srv.get("global_sample")),
            "val": {k: _f(val.get(k), 4) for k in ("macro_f1", "rare_recall", "missed_attack_rate")},
            "vote": {"messages": vote.get("messages"), "bytes": vote.get("bytes"), "ballots": vote.get("ballots"),
                     "ballot_bytes": vote.get("ballot_bytes")},
            "events": events, "rt": int(r.get("retransmits") or len(events)),
            "totals": {"bytes": tot.get("bytes") or (sum(u["bytes"] or 0 for u in up) + sum(d["bytes"] or 0 for d in down)),
                       "crypto_ms": _f(tot.get("crypto_ms"), 3)}}


def _series(rounds):
    out = []
    for r in rounds:
        val = (r.get("server") or {}).get("val") or {}
        tot = r.get("totals") or {}
        out.append({"r": int(r.get("round", 0)), "f1": _f(val.get("macro_f1"), 4),
                    "bytes": tot.get("bytes"), "ms": _f(tot.get("crypto_ms"), 3), "rt": int(r.get("retransmits") or 0)})
    return out


def _pick_vote_round(rounds, vote_start):
    """First voting round with a readmitted update (so the DQN step has something to weigh); else any voting round."""
    def voted(r):
        return any(x.get("yes", x.get("votes_yes")) is not None for x in (r.get("server") or {}).get("rows") or [])
    for want_readm in (True, False):
        for i, r in enumerate(rounds):
            srv = r.get("server") or {}
            if voted(r) and (not want_readm or srv.get("readmitted")):
                return i
    for i, r in enumerate(rounds):
        if vote_start is not None and r.get("round", 0) >= vote_start:
            return i
    return None


def _tamper(t):
    """Interception record (trace `tamper_demo`, or a run's event via _story) -> the page's shape. Older traces
    without the NACK / re-send part still show the rejection."""
    if not t:
        return None
    n, r = t.get("nack") or {}, t.get("resend") or {}
    return {"round": t.get("round"), "cid": t.get("cid"), "dir": t.get("dir") or "uplink", "error": t.get("error"),
            "before": t.get("cipher_hex_before") or "", "after": t.get("cipher_hex_after") or "",
            "byte": t.get("flipped_byte"), "bytes": t.get("bytes"), "nonce": t.get("nonce_hex_before") or "",
            "source": t.get("source"),
            "nack": n and {"bytes": n.get("bytes"), "hex": n.get("hex") or "", "nonce": n.get("nonce_hex") or "",
                           "body": n.get("body"), "tx": _f(n.get("transfer_s"), 4)},
            "resend": r and {"bytes": r.get("bytes"), "nonce": r.get("nonce_hex") or "", "ch": r.get("cipher_hex") or "",
                             "ok": bool(r.get("verify_ok")), "same": bool(r.get("bit_identical")),
                             "ps": _floats(r.get("plain_sample")), "ds": _floats(r.get("decrypted_sample")),
                             "attempts": r.get("attempts")}}


def _story(line):
    """A securelink.jsonl line with an uplink interception -> the record the interception scene shows (the same
    fields tools/make_demo_trace.py puts in the trace)."""
    for e in line.get("events") or []:
        if e.get("dir") != "uplink" or not e.get("recovered"):
            continue
        acc = next((u for u in line.get("uplink") or [] if u.get("cid") == e.get("cid")), {})
        return {"round": line.get("round"), "cid": e.get("cid"), "dir": "uplink", "error": e.get("error"),
                "flipped_byte": e.get("byte"), "bytes": e.get("bytes"), "nonce_hex_before": e.get("nonce_hex"),
                "cipher_hex_before": e.get("cipher_hex"), "cipher_hex_after": e.get("cipher_hex_rx"), "source": "run",
                "nack": {"bytes": e.get("nack_bytes"), "hex": e.get("nack_hex"), "nonce_hex": e.get("nack_nonce_hex"),
                         "body": e.get("nack_body"), "transfer_s": e.get("nack_transfer_s")},
                "resend": {"bytes": e.get("resend_bytes"), "nonce_hex": e.get("resend_nonce_hex"),
                           "cipher_hex": e.get("resend_cipher_hex"), "verify_ok": True,
                           "plain_sample": acc.get("plain_sample"), "decrypted_sample": acc.get("decrypted_sample"),
                           "bit_identical": bool(acc.get("plain_sample")) and acc.get("plain_sample") == acc.get("decrypted_sample"),
                           "attempts": acc.get("attempts")}}
    return None


def demo_payload(trace, dark=False):
    meta = trace.get("meta") or {}
    rounds = trace.get("rounds") or []
    if not rounds:
        return None
    vs = meta.get("vote_start")
    iv = _pick_vote_round(rounds, vs)
    tamper = trace.get("tamper_demo") or None
    it = None
    if tamper:
        it = next((i for i, r in enumerate(rounds) if r.get("round") == tamper.get("round")), len(rounds) - 1)
    keep = {"first": 0, "last": len(rounds) - 1}
    if iv is not None:
        keep["vote"] = iv
    if it is not None:
        keep["tamper"] = it
    crypto = meta.get("crypto") or {}
    return {"mode": "demo", "dark": bool(dark), "fixture": bool(meta.get("fixture")),
            "dataset": meta.get("dataset"), "method": meta.get("method"), "seed": meta.get("seed"),
            "rounds_total": meta.get("rounds") or len(rounds), "vote_start": vs, "rare": meta.get("rare_class"),
            "sites": _sites(meta.get("sites")), "server": SERVER.get("name"),
            "crypto": {k: v for k, v in crypto.items() if k != "enrollment"},
            "enroll": {str(e.get("cid")): e.get("fingerprint") for e in crypto.get("enrollment") or []},
            "R": {k: _round(rounds[i]) for k, i in keep.items()}, "series": _series(rounds),
            "info": {"K": meta.get("K"), "vfrac": meta.get("vote_start_frac"), "steps": meta.get("local_steps"),
                     "nc": meta.get("num_clients") or len(meta.get("sites") or []), "np": meta.get("num_poisoned"),
                     "tprob": meta.get("tamper_prob"), "maxret": meta.get("max_retries"),
                     "rt_total": sum(int(r.get("retransmits") or 0) for r in rounds), "run": meta.get("run_dir")},
            "tamper": _tamper(tamper)}


def _live_state(js):
    if not isinstance(js, dict):
        return None
    d = js.get("drones") or js.get("clients") or []
    if isinstance(d, dict):
        d = [dict(v, cid=int(k)) if isinstance(v, dict) else {"cid": int(k), "state": v} for k, v in d.items()]
    drones = []
    for x in d:
        if not isinstance(x, dict) or "cid" not in x:
            continue
        p = _f(x.get("progress"))
        drones.append({"cid": int(x["cid"]), "state": str(x.get("state") or x.get("status") or ""),
                       "progress": None if p is None else round(p, 2), "ms": _f(x.get("local_train_ms"), 1),
                       "dir": x.get("dir"), "attempt": x.get("attempt")})
    return {"round": js.get("round"), "phase": js.get("phase") or js.get("stage"), "drones": drones,
            "rt": js.get("retransmits") or 0, "event": js.get("event")}


def live_payload(meta, line, rec, live, series, dark=False, seed=None):
    sl = meta.get("secure_link") or {}
    cfg = meta.get("config") or {}
    R = meta.get("rounds") or (cfg.get("training") or {}).get("rounds")
    frac = (cfg.get("jiang") or {}).get("vote_start_frac")
    vs = int(round(frac * R)) if frac is not None and R else None
    n = ((cfg.get("clients") or {}).get("num_clients")) or None
    return {"mode": "live", "dark": bool(dark), "fixture": False, "rounds_total": R, "vote_start": vs,
            "sites": _sites(((meta.get("partitions") or {}).get(str(seed)) or {}).get("clients"), n),
            "server": SERVER.get("name"),
            "crypto": {k: v for k, v in sl.items() if k not in ("enrollment", "jobs")},
            "enroll": {str(e.get("cid")): e.get("fingerprint") for e in sl.get("enrollment") or []},
            "R": {"cur": _round(line, rec)} if line else {}, "series": series, "live": live,
            "tamper": _tamper(_story(line)) if line else None}


# ---------- page ----------
def _page(payload):
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    theme = "dark" if payload.get("dark") else "light"
    return (f'<!doctype html><html data-theme="{theme}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><style>{_CSS}</style></head>'
            f'<body>{_BODY}<script type="application/json" id="payload">{data}</script><script>{_JS}</script></body></html>')


def _dark():
    return bool(st.session_state.get("ui_dark"))


def _show(doc, height):
    if hasattr(st, "iframe"):  # st.iframe replaces the deprecated components.v1.html (same srcdoc iframe)
        st.iframe(doc, height=height, alt="Secure link: drones, encrypted links and the aggregation server")
    else:
        components.html(doc, height=height, scrolling=False)


def demo_html(trace, dark=False):
    p = demo_payload(trace, dark)
    return _page(p) if p else None


def render_demo(trace_path=DEMO_TRACE, height=780):
    """Full demo + step buttons from a real run's trace (tools/make_demo_trace.build): a trace dict, or a json path."""
    trace = trace_path if isinstance(trace_path, dict) else _load_json(trace_path)
    if not trace or not trace.get("rounds"):
        st.info(f"The demo trace `{trace_path}` has not been made yet. It is written by `tools/make_demo_trace.py` "
                "from the real mTLS run `demo_runs/mtls` (UAV-NIDD, Jiang + DQN).")
        return
    if (trace.get("meta") or {}).get("fixture"):
        st.warning("Showing a **development fixture** (synthetic values, genuine encryption), not the real demo "
                   "trace. Generate the real one with `tools/make_demo_trace.py`.")
    _show(demo_html(trace, _dark()), height)
    m = trace.get("meta") or {}
    st.caption(f"Values come from a real run: {m.get('dataset', '?')}, {m.get('method', '?')}, seed {m.get('seed', '?')}, "
               f"{m.get('rounds', '?')} rounds, voting from round {(m.get('vote_start') or 0) + 1}. Animation timing is "
               "compressed; the numbers (bytes, ms, simulated transfer time, hex) are the logged ones. "
               "Click a drone to inspect its packet in steps 3 and 5. Each step names the dashboard setting that runs it "
               "in a real training run.")


def _link_jobs(run_dir, meta):
    methods = meta.get("methods") or sorted(d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)))
    jobs = []
    for m in methods:
        mdir = os.path.join(run_dir, m)
        if not os.path.isdir(mdir):
            continue
        for d in sorted(os.listdir(mdir)):
            if d.startswith("seed_") and os.path.exists(os.path.join(mdir, d, LINK_LOG)):
                try:
                    jobs.append((m, int(d[5:])))
                except ValueError:
                    pass
    return jobs


def render_secure_link(run_dir, meta, height=720):
    """Live (while training) / replay view of a run trained with the encrypted link on."""
    meta = meta or {}
    jobs = _link_jobs(run_dir, meta) if run_dir and os.path.isdir(run_dir) else []
    if not jobs:
        st.info("No encrypted-link log in this run. The link log appears for runs trained from the dashboard (real mTLS, signed + encrypted updates). The **Demo** shows the whole link with real "
                "values from a short run.")
        return
    methods = list(dict.fromkeys(m for m, _ in jobs))
    run = os.path.basename(os.path.normpath(run_dir))
    c = st.columns([2, 1, 3])
    method = c[0].selectbox("Method", methods, key=f"sl_method_{run}", format_func=lambda m: m.replace("_", " "))
    seed = c[1].selectbox("Seed", [s for m, s in jobs if m == method], key=f"sl_seed_{run}")
    job = os.path.join(run_dir, method, f"seed_{seed}")
    live = meta.get("status") in ("running", "preparing") and os.path.exists(os.path.join(job, LIVE_STATE))
    if live:
        st.fragment(run_every=2.0)(_link_panel)(job, meta, run, True, height)
    else:
        _link_panel(job, meta, run, False, height)


def _link_panel(job, meta, run, live, height):
    lines = _jsonl(os.path.join(job, LINK_LOG))
    state = _live_state(_load_json(os.path.join(job, LIVE_STATE))) if live else None
    if not lines and not state:
        st.info("Waiting for the first round to finish…")
        return
    pick = None
    if lines:
        rounds = [int(x.get("round", i)) for i, x in enumerate(lines)]
        c = st.columns([1, 3])
        follow = c[0].toggle("Follow latest", value=True, key=f"sl_follow_{run}") if live else False
        if follow or len(rounds) == 1:
            pick = len(lines) - 1
        else:
            hit = [r for r, x in zip(rounds, lines) if x.get("retransmits")]
            r = c[1].select_slider("Round", options=rounds, value=rounds[-1], key=f"sl_round_{run}",
                                   format_func=lambda r: f"{r + 1}" + (" ⟲" if r in hit else ""),
                                   help="⟲ = a packet was rejected and re-sent in that round" if hit else None)
            pick = rounds.index(r)
    line = lines[pick] if pick is not None else None
    fl = _jsonl(os.path.join(job, ROUNDS))
    rec = None
    if line is not None:
        rec = next((x for x in reversed(fl) if x.get("round") == line.get("round")), None)
    f1 = {x.get("round"): (x.get("val") or {}).get("macro_f1") for x in fl}
    series = [{"r": int(x.get("round", i)), "f1": _f(f1.get(x.get("round")), 4), "bytes": (x.get("totals") or {}).get("bytes"),
               "ms": _f((x.get("totals") or {}).get("crypto_ms"), 3), "rt": int(x.get("retransmits") or 0)}
              for i, x in enumerate(lines)]
    ups = [u for x in lines for u in x.get("uplink") or []]
    bad = sum(1 for u in ups if not u.get("verify_ok", not u.get("error")))
    evs = [e for x in lines for e in x.get("events") or []]
    rt = sum(int(x.get("retransmits") or 0) for x in lines)
    m = st.columns(6)
    m[0].metric("Rounds logged", len(lines))
    m[1].metric("Link traffic", _fmt_bytes(sum(s["bytes"] or 0 for s in series)))
    m[2].metric("Crypto time", f"{sum(s['ms'] or 0 for s in series) / 1000:.2f} s")
    m[3].metric("Uplink packets verified", f"{len(ups) - bad}/{len(ups)}")
    m[4].metric("Vote copies sent", sum((x.get("vote") or {}).get("messages") or 0 for x in lines))
    m[5].metric("Re-sends (NACK)", rt, help="Packets rejected by the receiver (tampered on the link: InvalidTag) and "
                "re-sent after an encrypted re-send request. Only non-zero when interceptions are simulated "
                "(security.tamper_prob). "
                + (", ".join(f"{k}: {sum(1 for e in evs if e.get('dir') == k)}" for k in
                             ("uplink", "downlink", "vote", "ballot") if any(e.get("dir") == k for e in evs)) or ""))
    seed = os.path.basename(job).replace("seed_", "")
    _show(_page(live_payload(meta, line, rec, state, series, _dark(), seed)), height)


def _fmt_bytes(b):
    for unit, k in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if b >= k:
            return f"{b / k:.1f} {unit}"
    return f"{b} B"


# ---------- inline page: CSS, markup, script ----------
_CSS = r"""
:root{--bg:#ffffff;--panel:#f7f8fa;--panel2:#eef1f5;--line:rgba(49,51,63,.14);--line2:rgba(49,51,63,.26);
--ink:#31333f;--ink2:#4b5260;--muted:#6b7280;--faint:#a3abb6;--accent:#2563eb;--accent-soft:rgba(37,99,235,.10);
--ok:#3f9f4f;--ok-soft:rgba(63,159,79,.12);--bad:#d64545;--bad-soft:rgba(214,69,69,.10);--warn:#b98a06;
--warn-soft:rgba(201,154,6,.12);--ct:#a7aeb8;--ct-hot:#0b0d12;--icon-fill:#ffffff;--pkt:#ffffff;--shadow:rgba(15,23,42,.14);
--poison:#c62828;--rare:#2e7d32;--nack:#d97706;--nack-soft:rgba(217,119,6,.14);
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI","Source Sans Pro","Helvetica Neue",Arial,sans-serif}
html[data-theme=dark]{--bg:#0e1217;--panel:#131921;--panel2:#18202a;--line:rgba(150,165,185,.16);--line2:rgba(150,165,185,.30);
--ink:#d5dde6;--ink2:#b7c2cf;--muted:#8794a3;--faint:#566271;--accent:#4a8cf7;--accent-soft:rgba(74,140,247,.14);
--ok:#4fbf63;--ok-soft:rgba(79,191,99,.14);--bad:#ff6b6b;--bad-soft:rgba(255,107,107,.12);--warn:#e0b43a;
--warn-soft:rgba(224,180,58,.13);--ct:#5b6573;--ct-hot:#ffffff;--icon-fill:#161d26;--pkt:#1b232e;--shadow:rgba(0,0,0,.5);
--poison:#ef5350;--rare:#66bb6a;--nack:#f59e0b;--nack-soft:rgba(245,158,11,.16)}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 var(--sans);-webkit-font-smoothing:antialiased}
body{padding:2px 2px 0 2px;overflow:hidden;height:100vh;display:flex;flex-direction:column}
button{font:inherit;color:inherit}
.bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.btn{display:inline-flex;align-items:center;gap:6px;height:30px;padding:0 12px;border:1px solid var(--line2);background:var(--panel);
border-radius:6px;cursor:pointer;font-size:12.5px;font-weight:600;letter-spacing:.01em;transition:background .15s,border-color .15s}
.btn:hover{border-color:var(--accent)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.primary:hover{filter:brightness(1.08)}
.btn svg{width:12px;height:12px;fill:currentColor}
.btn.icon{padding:0 9px}
.seg{display:inline-flex;border:1px solid var(--line2);border-radius:6px;overflow:hidden;height:30px}
.seg button{border:0;background:var(--panel);padding:0 10px;cursor:pointer;font-size:12px;font-family:var(--mono);color:var(--muted)}
.seg button+button{border-left:1px solid var(--line)}
.seg button.on{background:var(--accent-soft);color:var(--accent);font-weight:700}
.prog{flex:1;min-width:120px;height:6px;background:var(--panel2);border-radius:3px;position:relative;cursor:pointer}
.prog .fill{position:absolute;left:0;top:0;bottom:0;background:var(--accent);border-radius:3px;width:0}
.prog .tick{position:absolute;top:-2px;width:1px;height:10px;background:var(--line2)}
.clock{font:12px var(--mono);color:var(--muted);min-width:92px;text-align:right}
.chips{display:flex;gap:3px;flex-wrap:wrap;margin-bottom:8px}
#chiptip{position:fixed;z-index:50;display:none;pointer-events:none;max-width:360px;background:var(--panel);color:var(--ink);
  border:1px solid var(--line);border-radius:4px;box-shadow:2px 2px 6px rgba(0,0,0,.18);padding:7px 9px;font-size:11.5px;line-height:1.4}
#chiptip b{font-weight:650}#chiptip .r{font:600 11px var(--mono);color:var(--accent,#1565c0);margin-bottom:3px}
#chiptip .m{color:var(--muted);margin-top:4px;font-size:10.5px}
.chip{border:1px solid var(--line);background:transparent;border-radius:999px;padding:2px 9px 2px 3px;font-size:11.5px;cursor:pointer;
display:inline-flex;align-items:center;gap:6px;color:var(--ink2);transition:all .15s}
.chip .n{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;background:var(--panel2);font:600 10.5px var(--mono);color:var(--muted)}
.chip:hover{border-color:var(--line2)}
.chip.on{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}
.chip.on .n{background:var(--accent);color:#fff}
.chip.done .n{background:var(--ok-soft);color:var(--ok)}
.main{display:grid;grid-template-columns:minmax(0,1.32fr) minmax(300px,1fr);gap:10px;flex:1;min-height:420px;padding-bottom:2px}
@media (max-width:820px){.main{grid-template-columns:1fr;flex:none}.stagewrap{height:520px}.insp{max-height:420px}body{overflow:auto;height:auto}}
.stagewrap{position:relative;border:1px solid var(--line);border-radius:8px;background:var(--panel);overflow:hidden;min-height:0;display:flex;flex-direction:column}
#stage{width:100%;flex:1;min-height:0;display:block}
.hud{position:absolute;left:12px;top:10px;display:flex;gap:6px;align-items:center;pointer-events:none}
.tag{font:600 10.5px var(--mono);letter-spacing:.06em;text-transform:uppercase;padding:3px 7px;border-radius:4px;
background:var(--bg);border:1px solid var(--line);color:var(--muted)}
.tag.acc{color:var(--accent);border-color:var(--accent-soft);background:var(--accent-soft)}
.tag.warn{color:var(--warn);background:var(--warn-soft);border-color:transparent}
.caption{margin:0 10px 10px;background:var(--bg);min-height:52px;
border:1px solid var(--line);border-radius:6px;padding:7px 11px;box-shadow:0 2px 10px var(--shadow);pointer-events:none}
.caption .k{font:600 10px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.caption .t{font-size:13px;color:var(--ink);margin-top:1px;min-height:19px}
.caption .how{font-size:11px;color:var(--muted);margin-top:3px;border-top:1px dashed var(--line);padding-top:3px}
.caption .how:empty{display:none}
.caption .how b{font-weight:650;color:var(--ink2)}
.role-poisoned{color:var(--poison)!important;font-weight:650}.role-rare{color:var(--rare)!important;font-weight:650}
text.role-poisoned{fill:var(--poison)!important}text.role-rare{fill:var(--rare)!important}
.badge.nack{background:var(--nack-soft);color:var(--nack)}
.nackc{color:var(--nack)}
.step.nack .d{border-color:var(--nack);color:var(--nack);background:var(--nack-soft)}
.legend{position:absolute;right:12px;top:10px;font-size:10.5px;color:var(--muted);display:grid;grid-template-columns:auto auto;
gap:2px 8px;align-items:center;background:color-mix(in srgb,var(--panel) 85%,transparent);padding:4px 6px;border-radius:5px}
.legend svg{display:block}
.insp{border:1px solid var(--line);border-radius:8px;background:var(--panel);overflow:auto;padding:12px 14px;min-height:0}
.insp h3{margin:0 0 2px 0;font-size:14px;font-weight:650;letter-spacing:.005em}
.insp .sub{color:var(--muted);font-size:12px;margin-bottom:10px}
.insp h4{margin:12px 0 5px;font:600 10.5px var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.rv{opacity:0;transform:translateY(3px);transition:opacity .35s ease,transform .35s ease}
.rv.on{opacity:1;transform:none}
table.t{border-collapse:collapse;width:100%;font:11.5px var(--mono)}
table.t th{text-align:left;font:600 9.5px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
border-bottom:1px solid var(--line2);padding:3px 5px;white-space:nowrap}
table.t td{padding:3px 5px;border-bottom:1px solid var(--line);white-space:nowrap}
table.t td.r,table.t th.r{text-align:right}
table.t tr.hl td{background:var(--accent-soft)}
.m{font-family:var(--mono)}
.dim{color:var(--muted)}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warnc{color:var(--warn)}
.badge{display:inline-flex;align-items:center;gap:5px;font:700 11px var(--mono);letter-spacing:.03em;padding:3px 8px;border-radius:4px}
.badge.ok{background:var(--ok-soft);color:var(--ok)}
.badge.bad{background:var(--bad-soft);color:var(--bad)}
.badge.acc{background:var(--accent-soft);color:var(--accent)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:12px}
.kv .k{color:var(--muted)}
.suite{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0}
.suite span{font:600 11px var(--mono);padding:3px 7px;border:1px solid var(--line2);border-radius:4px}
.suite i{font-style:normal;color:var(--faint);align-self:center}
.vals{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:4px}
.vals div{background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:3px 5px;font:11px var(--mono);overflow:hidden}
.vals .v{color:var(--ink)}
.vals .h{color:var(--muted);font-size:10.5px;letter-spacing:.03em}
.vals div:hover{border-color:var(--accent)}
.hex{font:11px/1.6 var(--mono);word-break:break-all;letter-spacing:.02em}
.pk{border:1px dashed var(--line2);border-radius:6px;padding:7px 9px;background:var(--bg);cursor:default;transition:border-color .2s}
.pk .f{display:grid;grid-template-columns:104px 1fr;gap:6px;align-items:baseline}
.pk .f+.f{margin-top:3px}
.pk .lbl{font:600 9.5px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--faint);transition:color .2s}
.pk .ct{color:var(--ct);transition:color .25s}
.pk:hover{border-color:var(--ink2);border-style:solid}
.pk:hover .ct,.pk.hot .ct{color:var(--ct-hot)}
.pk:hover .lbl.n{color:var(--accent)} .pk:hover .lbl.g{color:var(--ok)} .pk:hover .lbl.s{color:var(--warn)} .pk:hover .lbl.c{color:var(--ink)}
.pk .note{font-size:10.5px;color:var(--faint);margin-top:4px}
.anat{display:flex;height:16px;border-radius:3px;overflow:hidden;margin:4px 0 2px;font:600 9px var(--mono);letter-spacing:.04em}
.anat span{display:grid;place-items:center;color:#fff;white-space:nowrap;overflow:hidden}
.anat .a{background:#7c8796}.anat .n{background:#3b6fd8}.anat .c{background:#5d6673}.anat .g{background:#3f9f4f}.anat .s{background:#b98a06}
.flipped{background:var(--bad);color:#fff;border-radius:2px;padding:0 1px}
.cmp td.eq{color:var(--ok)}
.bars .b{display:grid;grid-template-columns:62px 1fr 64px;gap:8px;align-items:center;font:11.5px var(--mono);margin:3px 0}
.bars .tr{height:10px;background:var(--panel2);border-radius:2px;position:relative;overflow:hidden}
.bars .fl{position:absolute;left:0;top:0;bottom:0;border-radius:2px;background:var(--faint);transition:width .5s ease}
.bars .fl.k{background:var(--accent)}.bars .fl.h{background:var(--warn)}.bars .fl.okb{background:var(--ok)}.bars .fl.badb{background:var(--bad)}
.cut{border-top:1px dashed var(--line2);margin:5px 0;position:relative}
.cut span{position:absolute;right:0;top:-8px;background:var(--panel);padding:0 4px;font:600 9.5px var(--mono);color:var(--muted);letter-spacing:.05em;text-transform:uppercase}
.step{display:grid;grid-template-columns:20px 1fr;gap:8px;margin:7px 0}
.step .d{width:18px;height:18px;border-radius:50%;border:1px solid var(--line2);display:grid;place-items:center;font:600 10px var(--mono);color:var(--muted)}
.step.done .d{border-color:var(--ok);color:var(--ok);background:var(--ok-soft)}
.step.fail .d{border-color:var(--bad);color:var(--bad);background:var(--bad-soft)}
.step b{font-weight:650}
.note{font-size:11.5px;color:var(--muted);margin-top:8px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.metrics div{border:1px solid var(--line);border-radius:5px;padding:5px 7px;background:var(--bg)}
.metrics .k{font:600 9.5px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.metrics .v{font:600 16px var(--mono)}
.spark{width:100%;height:auto;display:block}
/* stage */
.ringg{fill:none;stroke:var(--line);stroke-width:1}
.link{stroke:var(--line2);stroke-width:1.2;fill:none;transition:stroke .3s,opacity .3s}
.link.off{stroke-dasharray:3 4;opacity:.6}
.link.sec{stroke:var(--line2)}
.link.sel{stroke:var(--accent)}
.link.held{stroke:var(--warn);stroke-dasharray:5 4}
.link.back{stroke:var(--ok)}
.link.rej{stroke:var(--bad);stroke-dasharray:2 4;opacity:.55}
.link.faint{opacity:.3}
.link.resend{stroke:var(--nack);stroke-dasharray:4 3;stroke-width:1.8}
.lk .bgc{fill:var(--panel);stroke:var(--line2);stroke-width:.8}
.glyph{fill:none;stroke:var(--muted);stroke-width:1.1;stroke-linecap:round;stroke-linejoin:round}
.dn{cursor:pointer;transition:opacity .35s}
.dn .body{fill:var(--icon-fill);stroke:var(--ink);stroke-width:1.3}
.dn .arm{stroke:var(--ink);stroke-width:2.2;stroke-linecap:round}
.dn .rot{fill:var(--icon-fill);stroke:var(--ink);stroke-width:1.1}
.dn .hub{fill:var(--ink)}
.dn .blade{stroke:var(--muted);stroke-width:1.2;stroke-linecap:round}
.dn .cam{fill:var(--accent)}
.dn .lab{font:650 11px var(--sans);fill:var(--ink);text-anchor:middle}
.dn .site{font:10px var(--sans);fill:var(--muted);text-anchor:middle}
.dn .st{font:10px var(--mono);fill:var(--muted);text-anchor:middle}
.dn .st.ok{fill:var(--ok)}.dn .st.bad{fill:var(--bad)}.dn .st.acc{fill:var(--accent)}.dn .st.warn{fill:var(--warn)}.dn .st.nack{fill:var(--nack);font-weight:700}
.dn .track{fill:none;stroke:var(--line);stroke-width:2}
.dn .ring{fill:none;stroke:var(--accent);stroke-width:2;stroke-linecap:round;opacity:0}
.dn .halo{fill:var(--accent-soft);stroke:var(--accent);stroke-width:1;stroke-dasharray:2 3;opacity:0;transition:opacity .3s}
.dn.focus .halo{opacity:1}
.dn .cert circle{fill:var(--ok)}.dn .cert path{stroke:#fff;stroke-width:1.4;fill:none;stroke-linecap:round;stroke-linejoin:round}
.srv .rack{fill:var(--icon-fill);stroke:var(--ink);stroke-width:1.4}
.srv .unit{fill:var(--panel2);stroke:var(--ink);stroke-width:.9}
.srv .bay{fill:none;stroke:var(--muted);stroke-width:.8}
.srv .led{fill:var(--ok)}.srv .led2{fill:var(--accent);opacity:.7}
.srv .foot{stroke:var(--ink);stroke-width:1.4;stroke-linecap:round}
.srv .plate{fill:none;stroke:var(--accent);stroke-width:1.2;opacity:0}
.srv.flash .plate{opacity:.9}
.srv.alarm .rack{stroke:var(--bad)} .srv.alarm .plate{stroke:var(--bad);opacity:.9}
.srvlab{font:650 11.5px var(--sans);fill:var(--ink);text-anchor:middle}
.srvsub{font:10px var(--mono);fill:var(--muted);text-anchor:middle}
.slot circle{fill:var(--panel);stroke:var(--line2);stroke-width:1}
.slot path{fill:none;stroke:#fff;stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round;opacity:0}
.slot.recv circle{fill:var(--panel);stroke:var(--ok)} .slot.recv path.c{stroke:var(--ok);opacity:1}
.slot.ok circle{fill:var(--ok);stroke:var(--ok)} .slot.ok path.c{opacity:1}
.slot.bad circle{fill:var(--bad);stroke:var(--bad)} .slot.bad path.x{opacity:1}
.slot.sel circle{stroke:var(--accent);fill:var(--accent)} .slot.sel path.c{opacity:1;stroke:#fff}
.slot.held circle{stroke:var(--warn);fill:var(--warn-soft)}
.slot.nack circle{stroke:var(--nack);fill:var(--nack-soft)}
.ca .doc{fill:var(--icon-fill);stroke:var(--ink);stroke-width:1.1}
.ca .ln{stroke:var(--muted);stroke-width:1}
.ca .seal{fill:var(--ok)}
.ca text{font:10px var(--sans);fill:var(--muted);text-anchor:middle}
.pkt rect{stroke-width:1.1;fill:var(--pkt)}
.pkt.up rect{stroke:var(--ink2)}
.pkt.down rect{stroke:var(--accent)} .pkt.down .glyph{stroke:var(--accent)}
.pkt.vote rect{stroke:var(--warn)} .pkt.vote .glyph{stroke:var(--warn)}
.pkt.ballot rect{stroke:var(--muted)}
.pkt.bad rect{stroke:var(--bad);fill:var(--bad-soft)} .pkt.bad .glyph{stroke:var(--bad)}
.pkt.nack rect{stroke:var(--nack);fill:var(--nack-soft);stroke-width:1.3} .pkt.nack text{font:700 6.5px var(--mono);fill:var(--nack);text-anchor:middle}
.pkt.resend rect{stroke:var(--ok);stroke-width:1.4} .pkt.resend .glyph{stroke:var(--ok)}
.pkt .dash{stroke:var(--faint);stroke-width:1;stroke-linecap:round}
.pkt.key circle{fill:var(--accent)}
.pkt.cert rect{stroke:var(--ok)}
.dot{fill:var(--accent)}
.mitm circle{fill:var(--bad-soft);stroke:var(--bad);stroke-width:1.2}
.mitm line{stroke:var(--bad);stroke-width:1.1}
.mitm text{font:600 10px var(--mono);fill:var(--bad);text-anchor:middle}
.beam{stroke:var(--bad);stroke-width:1;stroke-dasharray:2 2}
.alarmtxt rect{fill:var(--bad)} .alarmtxt text{fill:#fff;font:700 10.5px var(--mono);text-anchor:middle;letter-spacing:.04em}
"""

_BODY = r"""
<div class="bar">
  <button class="btn primary" id="playall"><svg viewBox="0 0 12 12"><path d="M2.5 1.5v9l8-4.5z"/></svg><span>Play full demo</span></button>
  <button class="btn icon" id="pause" title="Pause / resume"><svg viewBox="0 0 12 12" id="pauseic"><rect x="2.5" y="1.5" width="2.6" height="9" rx=".6"/><rect x="6.9" y="1.5" width="2.6" height="9" rx=".6"/></svg></button>
  <button class="btn icon" id="prev" title="Previous step"><svg viewBox="0 0 12 12"><path d="M8.5 1.8 3.6 6l4.9 4.2z"/></svg></button>
  <button class="btn" id="next" title="Next step"><span>Step</span><svg viewBox="0 0 12 12"><path d="M3.5 1.8 8.4 6l-4.9 4.2z"/></svg></button>
  <div class="seg" id="speed"><button data-s="0.5">0.5×</button><button data-s="1" class="on">1×</button><button data-s="2">2×</button></div>
  <div class="prog" id="prog"><div class="fill"></div></div>
  <span class="clock" id="clock">0:00 / 0:00</span>
</div>
<div class="chips" id="chips"></div>
<div class="main">
  <div class="stagewrap">
    <svg id="stage" viewBox="0 0 720 600" preserveAspectRatio="xMidYMid meet" aria-label="Drones, links and the aggregation server"></svg>
    <div class="hud"><span class="tag" id="rtag">Round –</span><span class="tag warn" id="vtag" style="display:none">voting active</span><span class="tag acc" id="ftag" style="display:none">fixture</span></div>
    <div class="legend" id="legend"></div>
    <div class="caption"><div class="k" id="capk"></div><div class="t" id="capt"></div><div class="how" id="caph"></div></div>
  </div>
  <aside class="insp" id="insp"></aside>
</div>
"""

_JS = r"""
(function(){
"use strict";
const D = JSON.parse(document.getElementById('payload').textContent);
const NS = 'http://www.w3.org/2000/svg';
const $ = s => document.querySelector(s);
const svg = $('#stage'), insp = $('#insp');
function S(tag, a, p){ const e = document.createElementNS(NS, tag); if (a) for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; }
const clamp = (x, a=0, b=1) => Math.max(a, Math.min(b, x));
const ease = x => x < .5 ? 2*x*x : 1 - Math.pow(-2*x + 2, 2)/2;
const seg = (t, a, b) => clamp((t - a)/(b - a));
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pad2 = n => String(n).padStart(2, '0');
const LBL = {}; D.sites.forEach(s => LBL[s.cid] = s.label);
const droneName = c => LBL[c] || ('Drone ' + pad2(c + 1));
const fmtF = x => x == null ? '–' : (x < 0 ? '−' : '+') + Math.abs(x).toFixed(7);
const fmtInt = n => n == null ? '–' : Number(n).toLocaleString('en-US');
const fmtB = b => b == null ? '–' : b >= 1048576 ? (b/1048576).toFixed(2) + ' MB' : b >= 1024 ? (b/1024).toFixed(1) + ' KB' : b + ' B';
const fmtMs = x => x == null ? '–' : x < 1 ? x.toFixed(3) + ' ms' : x < 100 ? x.toFixed(2) + ' ms' : x.toFixed(0) + ' ms';
const fmtS = x => x == null ? '–' : (x*1000).toFixed(1) + ' ms';
const words = (h, n) => (h.match(/.{1,8}/g) || []).slice(0, n).join(' ');
const shortHex = (h, a=16, b=8) => h.length > a + b + 2 ? h.slice(0, a) + '…' + h.slice(-b) : h;
function f32hex(x){ const b = new DataView(new ArrayBuffer(4)); b.setFloat32(0, x, true); let s = ''; for (let i = 0; i < 4; i++) s += b.getUint8(i).toString(16).padStart(2, '0'); return s; }
const byCid = arr => { const m = {}; (arr || []).forEach(x => m[x.cid] = x); return m; };
const sites = D.sites, N = sites.length, cids = sites.map(s => s.cid);
const RT = D.rounds_total || '?';
if (D.fixture) $('#ftag').style.display = '';

// ---------- geometry ----------
const W = 720, H = 600;
// sites are in km east / north of the server (0, 0): one uniform scale, so the layout matches the site map
const KX = sites.map(s => s.x).concat([0]), KY = sites.map(s => s.y).concat([0]);
const x0 = Math.min(...KX), x1 = Math.max(...KX), y0 = Math.min(...KY), y1 = Math.max(...KY);
const K = Math.min((W - 140)/((x1 - x0) || 1), (H - 142)/((y1 - y0) || 1));
const OX = (W - (x1 - x0)*K)/2 - x0*K, OY = 68 + (H - 142 - (y1 - y0)*K)/2;
const P = s => ({x: OX + s.x*K, y: OY + (y1 - s.y)*K});
const SV = P({x: 0, y: 0});
const pos = {}; sites.forEach(s => pos[s.cid] = P(s));
const links = {};
sites.forEach(s => {
  const p = pos[s.cid], dx = SV.x - p.x, dy = SV.y - p.y, L = Math.hypot(dx, dy) || 1, ux = dx/L, uy = dy/L;
  let ang = Math.atan2(dy, dx)*180/Math.PI; if (ang > 90) ang -= 180; if (ang < -90) ang += 180;
  links[s.cid] = {A: {x: p.x + ux*31, y: p.y + uy*31}, B: {x: SV.x - ux*72, y: SV.y - uy*72},
                  S: {x: SV.x - ux*62, y: SV.y - uy*62}, ang, ux, uy};
});
const lerp = (a, b, f) => ({x: a.x + (b.x - a.x)*f, y: a.y + (b.y - a.y)*f});

// ---------- drawing ----------
const defs = S('defs', null, svg);
const fl = S('filter', {id: 'sh', x: '-30%', y: '-30%', width: '160%', height: '160%'}, defs);
S('feDropShadow', {dx: 0, dy: 1, stdDeviation: 1.6, 'flood-color': getComputedStyle(document.documentElement).getPropertyValue('--shadow').trim() || 'rgba(0,0,0,.2)'}, fl);
const gGrid = S('g', null, svg), gLinks = S('g', null, svg), gSlots = S('g', null, svg), gNodes = S('g', null, svg),
      gSrv = S('g', null, svg), gPk = S('g', null, svg), gTop = S('g', null, svg);
[110, 190, 270].forEach((r, i) => S('circle', {cx: SV.x, cy: SV.y, r, class: 'ringg', 'stroke-dasharray': i ? '2 5' : '', opacity: .8}, gGrid));

function lockGlyph(p, x, y, k=1){
  const g = S('g', {transform: `translate(${x},${y}) scale(${k})`, class: 'glyph'}, p);
  S('path', {d: 'M-2.4 -0.6 V-2.3 A2.4 2.4 0 0 1 2.4 -2.3 V-0.6'}, g);
  S('rect', {x: -3.3, y: -0.8, width: 6.6, height: 4.8, rx: 1}, g);
  return g;
}
function droneIcon(p){
  const g = S('g', null, p), blades = [];  // no SVG filter: blades and opacity animate, filters re-rasterise
  S('line', {x1: -13, y1: -13, x2: 13, y2: 13, class: 'arm'}, g);
  S('line', {x1: 13, y1: -13, x2: -13, y2: 13, class: 'arm'}, g);
  [[-14, -14], [14, -14], [14, 14], [-14, 14]].forEach(([x, y]) => {
    S('circle', {cx: x, cy: y, r: 8, class: 'rot'}, g);
    const b = S('g', {transform: `translate(${x},${y})`}, g);
    const bl = S('g', null, b);
    S('line', {x1: -6, y1: 0, x2: 6, y2: 0, class: 'blade'}, bl);
    S('circle', {r: 1.4, class: 'hub'}, b);
    blades.push(bl);
  });
  S('rect', {x: -7.5, y: -9, width: 15, height: 18, rx: 4.5, class: 'body'}, g);
  S('circle', {cx: 0, cy: -5, r: 1.7, class: 'cam'}, g);
  S('line', {x1: -3, y1: 3, x2: 3, y2: 3, class: 'blade'}, g);
  return blades;
}
function serverIcon(p){
  S('rect', {x: -36, y: -46, width: 72, height: 92, rx: 7, class: 'plate'}, p);
  const g = S('g', {filter: 'url(#sh)'}, p);
  S('rect', {x: -28, y: -40, width: 56, height: 80, rx: 4, class: 'rack'}, g);
  for (let i = 0; i < 3; i++){
    const y = -33 + i*23.5;
    S('rect', {x: -22, y, width: 44, height: 18, rx: 2, class: 'unit'}, g);
    for (let j = 0; j < 4; j++) S('rect', {x: -18 + j*5.6, y: y + 4, width: 3.4, height: 10, rx: .8, class: 'bay'}, g);
    S('circle', {cx: 11, cy: y + 9, r: 1.7, class: 'led'}, g);
    S('circle', {cx: 16.5, cy: y + 9, r: 1.7, class: 'led2'}, g);
  }
  S('line', {x1: -20, y1: 44, x2: -12, y2: 44, class: 'foot'}, g);
  S('line', {x1: 12, y1: 44, x2: 20, y2: 44, class: 'foot'}, g);
}
// links + mid locks
const LN = {}, LK = {};
cids.forEach(c => {
  const L = links[c];
  LN[c] = S('line', {x1: L.A.x, y1: L.A.y, x2: L.B.x, y2: L.B.y, class: 'link off'}, gLinks);
  const m = lerp(L.A, L.B, .5), g = S('g', {class: 'lk', transform: `translate(${m.x},${m.y})`, opacity: 0}, gLinks);
  S('circle', {r: 7, class: 'bgc'}, g); lockGlyph(g, 0, .6, .85);
  LK[c] = g;
});
// receipt slots round the server
const SL = {};
cids.forEach(c => {
  const s = links[c].S, g = S('g', {class: 'slot', transform: `translate(${s.x},${s.y})`}, gSlots);
  S('circle', {r: 5.5}, g);
  S('path', {d: 'M-2.6 0.1 L-0.8 1.9 L2.7 -1.9', class: 'c'}, g);
  S('path', {d: 'M-2 -2 L2 2 M2 -2 L-2 2', class: 'x'}, g);
  SL[c] = g;
});
// server
const srv = S('g', {class: 'srv', transform: `translate(${SV.x},${SV.y})`}, gSrv);
serverIcon(srv);
S('text', {x: SV.x, y: SV.y + 86, class: 'srvlab'}, gSrv).textContent = D.server || 'Aggregation server';
const srvSub = S('text', {x: SV.x, y: SV.y + 99, class: 'srvsub'}, gSrv);
// CA
const CA = {x: SV.x + 86, y: SV.y - 80};
const ca = S('g', {class: 'ca', transform: `translate(${CA.x},${CA.y})`, opacity: 0}, gSrv);
S('rect', {x: -13, y: -10, width: 26, height: 19, rx: 2, class: 'doc'}, ca);
S('line', {x1: -9, y1: -5, x2: 6, y2: -5, class: 'ln'}, ca); S('line', {x1: -9, y1: -1, x2: 2, y2: -1, class: 'ln'}, ca);
S('path', {d: 'M5 6 l-2 7 l3 -2 l3 2 l-2 -7z', class: 'seal'}, ca); S('circle', {cx: 6, cy: 4, r: 4, class: 'seal'}, ca);
S('text', {y: 24}, ca).textContent = 'Operator CA';
// drones
const DN = {};
D.crypto = D.crypto || {};
sites.forEach(s => {
  const p = pos[s.cid], g = S('g', {class: 'dn', transform: `translate(${p.x},${p.y})`}, gNodes);
  S('circle', {r: 30, class: 'halo'}, g);
  S('circle', {r: 26, class: 'track', opacity: 0}, g);
  const ring = S('circle', {r: 26, class: 'ring', transform: 'rotate(-90)', 'stroke-dasharray': `0 ${2*Math.PI*26}`}, g);
  const blades = droneIcon(g);
  const cert = S('g', {class: 'cert', transform: 'translate(19,-19)', opacity: 0}, g);
  S('circle', {r: 5.5}, cert); S('path', {d: 'M-2.4 0.1 L-0.7 1.8 L2.5 -1.7'}, cert);
  // labels on the outer side, away from the link, so packets never cross them
  const above = p.y < SV.y - 60, y0 = above ? -62 : 42;
  // poisoned drones red, the rare drone green (role from the site's own record; the server never uses it)
  const rc = (s.role === 'poisoned' || s.role === 'rare') ? ' role-' + s.role : '';
  S('text', {y: y0, class: 'lab' + rc}, g).textContent = droneName(s.cid);
  S('text', {y: y0 + 12, class: 'site' + rc}, g).textContent = s.name;
  const st = S('text', {y: y0 + 25, class: 'st'}, g);
  g.addEventListener('click', () => setFocus(s.cid));
  const t = S('title', null, g); t.textContent = droneName(s.cid) + ' · ' + s.name;
  DN[s.cid] = {g, ring, track: g.querySelector('.track'), blades, cert, st, _: {}};
});
const top = S('g', null, gTop);

// state setters (only touch the DOM when a value changes)
function setA(o, el, k, v){ const key = k + '@' + (el.__id || (el.__id = Math.random())); if (o[key] === v) return; o[key] = v; el.setAttribute(k, v); }
const cache = {};
function drone(c, o){
  const d = DN[c]; if (!d) return;
  if (o.op != null) setA(cache, d.g, 'opacity', o.op);
  if (o.ring !== undefined){
    const C = 2*Math.PI*26, v = o.ring == null ? 0 : clamp(o.ring);
    setA(cache, d.ring, 'stroke-dasharray', `${(C*v).toFixed(1)} ${C}`);
    setA(cache, d.ring, 'opacity', o.ring == null || v <= 0 ? 0 : .85);
    setA(cache, d.track, 'opacity', o.ring == null ? 0 : 1);
  }
  if (o.st !== undefined){ if (d.st.textContent !== o.st) d.st.textContent = o.st; setA(cache, d.st, 'class', 'st ' + (o.cls || '')); }
  if (o.cert != null) setA(cache, d.cert, 'opacity', o.cert);
  if (o.spin != null) d.blades.forEach((b, i) => b.setAttribute('transform', `rotate(${(o.spin*(i % 2 ? -1 : 1)) % 360})`));
}
function link(c, cls, w, op){
  setA(cache, LN[c], 'class', 'link ' + cls);
  setA(cache, LN[c], 'stroke-width', w == null ? 1.2 : w);
  setA(cache, LK[c], 'opacity', op == null ? (cls.includes('off') ? 0 : 1) : op);
}
function slot(c, cls){ setA(cache, SL[c], 'class', 'slot ' + (cls || '')); }
let focus = null;
function setFocus(c){
  focus = c; Object.keys(DN).forEach(k => DN[k].g.classList.toggle('focus', +k === c));
  const sc = scenes[cur]; if (sc && sc.usesFocus){ enter(cur, playing); }
}
function packet(kind, title){
  const g = S('g', {class: 'pkt ' + kind, style: 'display:none'}, gPk);
  if (kind === 'nack'){ S('rect', {x: -10, y: -5, width: 20, height: 10, rx: 2}, g); S('text', {y: 2.3}, g).textContent = 'NACK'; if (title){ S('title', null, g).textContent = title; } return g; }
  if (kind === 'key'){ S('circle', {r: 3.2}, g); return g; }
  if (kind === 'cert'){ S('rect', {x: -7, y: -5, width: 14, height: 10, rx: 1.5}, g); S('line', {x1: -4, y1: -1.5, x2: 4, y2: -1.5, class: 'dash'}, g); S('line', {x1: -4, y1: 1.5, x2: 2, y2: 1.5, class: 'dash'}, g); return g; }
  if (kind === 'ballot'){ S('rect', {x: -6, y: -4, width: 12, height: 8, rx: 4}, g); return g; }
  S('rect', {x: -13, y: -5.5, width: 26, height: 11, rx: 5.5}, g);
  lockGlyph(g, -6.2, -0.4, .72);
  S('line', {x1: -0.5, y1: -1.6, x2: 8, y2: -1.6, class: 'dash'}, g);
  S('line', {x1: -0.5, y1: 1.6, x2: 5, y2: 1.6, class: 'dash'}, g);
  if (title){ const t = S('title', null, g); t.textContent = title; }
  return g;
}
function place(g, a, b, f, ang){
  if (f <= 0 || f >= 1){ if (g.style.display !== 'none') g.style.display = 'none'; return; }
  if (g.style.display === 'none') g.style.display = '';
  const p = lerp(a, b, ease(f));
  g.setAttribute('transform', `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})` + (ang != null ? ` rotate(${ang.toFixed(1)})` : ''));
}
function show(g, at, ang){ g.style.display = ''; g.setAttribute('transform', `translate(${at.x.toFixed(1)},${at.y.toFixed(1)})` + (ang != null ? ` rotate(${ang.toFixed(1)})` : '')); }
function hide(g){ g.style.display = 'none'; }
const up = (g, c, f) => place(g, links[c].A, links[c].B, f, links[c].ang);
const down = (g, c, f) => place(g, links[c].B, links[c].A, f, links[c].ang);

// ---------- legend ----------
(function(){
  const lg = $('#legend'), items = [['sec', 'encrypted link'], ['sel', 'kept by filter'], ['held', 'held back'], ['back', 'voted back in'], ['rej', 'voted out']];
  let h = '';
  items.forEach(([c, t]) => { h += `<svg width="22" height="8"><line x1="1" y1="4" x2="21" y2="4" class="link ${c}" stroke-width="1.6"/></svg><span>${t}</span>`; });
  h += `<svg width="22" height="10" viewBox="-11 -5 22 10"><rect x="-10" y="-4" width="20" height="8" rx="4" fill="none" stroke="currentColor" stroke-width="1"/></svg><span>packet (locked)</span>`;
  lg.innerHTML = h;
})();

// ---------- round data ----------
function prep(R){
  if (!R) return null;
  R.L = byCid(R.local); R.U = byCid(R.up); R.Dn = byCid(R.down); R.W = byCid(R.rows);
  R.voted = R.rows.filter(x => x.yes != null);
  return R;
}
const RR = {}; Object.keys(D.R || {}).forEach(k => RR[k] = prep(D.R[k]));
const R0 = RR.first || RR.cur, RV = RR.vote, RL = RR.last, RTm = RR.tamper;
const vs = D.vote_start;
const rnum = R => R ? R.round + 1 : '–';
function hud(R, label){
  $('#rtag').textContent = label || ('Round ' + rnum(R) + ' / ' + RT);
  $('#vtag').style.display = (R && vs != null && R.round >= vs) ? '' : 'none';
}
function defaultFocus(R){
  if (focus != null && R && R.U[focus]) return focus;
  const u = R && R.up.find(u => u.ps && u.ps.length);
  return u ? u.cid : cids[0];
}
const enrolled = c => D.enroll && D.enroll[String(c)];

// steady state: every link encrypted, every drone idle at full opacity
function base(o = {}){
  gPk.innerHTML = ''; top.innerHTML = '';
  srv.classList.remove('flash', 'alarm');
  cids.forEach(c => DN[c].g.classList.remove('focus'));
  srvSub.textContent = o.sub || '';
  ca.setAttribute('opacity', o.ca || 0);
  cids.forEach(c => {
    link(c, o.link || 'sec');
    slot(c, '');
    drone(c, {op: 1, ring: null, st: o.st || '', cls: '', cert: 0, spin: 0});
  });
}

// ---------- inspector helpers ----------
function panel(title, sub, body){ insp.innerHTML = `<h3>${title}</h3><div class="sub">${sub || ''}</div>${body}`; insp.scrollTop = 0; }
const rv = (at, html, tag='div', cls='') => `<${tag} class="rv ${cls}" data-at="${at}">${html}</${tag}>`;
function reveal(t){
  let last = null;
  insp.querySelectorAll('[data-at]').forEach(e => { const on = t >= +e.dataset.at; if (on && !e.classList.contains('on')) last = e; e.classList.toggle('on', on); });
  if (last && playing){  // keep what was just revealed in view (the inspector scrolls; the stage does not)
    const r = last.getBoundingClientRect(), p = insp.getBoundingClientRect();
    if (r.bottom > p.bottom - 4) insp.scrollTop += Math.max(0, Math.min(r.bottom - p.bottom + 12, r.top - p.top - 8));
  }
}
function suite(){
  const c = D.crypto || {};
  return `<div class="suite"><span>${esc(c.kex || 'X25519')}</span><i>→</i><span>${esc(c.kdf || 'HKDF-SHA256')}</span><i>→</i><span>${esc(c.aead || 'AES-256-GCM')}</span><i>+</i><span>${esc(c.sig || 'Ed25519')}</span></div>`;
}
function valsGrid(vals, hex){
  return `<div class="vals">` + vals.map((v, i) => `<div><div class="v">${fmtF(v)}</div>${hex ? `<div class="h">${hex[i] || ''}</div>` : ''}</div>`).join('') + `</div>`;
}
function hexf32(h){ const b = new DataView(new ArrayBuffer(4)); for (let i = 0; i < 4; i++) b.setUint8(i, parseInt(h.substr(2*i, 2), 16)); return b.getFloat32(0, true); }
// The plaintext bytes line up with the logged values when each 4-byte word decodes to that value (the demo trace
// rounds the samples to 6 decimals, the hex is exact).
function plainWords(u){
  const w = (u.ph.match(/.{8}/g) || []).slice(0, u.ps.length);
  const aligned = w.length === u.ps.length && u.ps.every((x, i) => Math.abs(hexf32(w[i]) - x) <= 1e-6 + 1e-6*Math.abs(x));
  return {w, aligned};
}
function packetBlock(u, R, hot){
  const hdr = `{"cid":${u.cid},"dir":"uplink","round":${R.round}}`;  // the AAD exactly as sealed (sorted keys)
  return `<div class="pk${hot ? ' hot' : ''}">
   <div class="f"><span class="lbl a">header · AAD</span><span class="hex ct">${esc(hdr)}</span></div>
   <div class="f"><span class="lbl n">nonce · 12 B</span><span class="hex ct">${esc(u.nonce)}</span></div>
   <div class="f"><span class="lbl c">ciphertext</span><span class="hex ct">${esc(words(u.ch, 12))} …</span></div>
   <div class="f"><span class="lbl g">GCM tag · 16 B</span><span class="hex ct">${esc(u.tag)}</span></div>
   <div class="f"><span class="lbl s">signature · 64 B</span><span class="hex ct">${esc(shortHex(u.sig, 32, 16))}</span></div>
   <div class="note">first 48 B of ${fmtInt(u.bytes)} B shown · hover to read · "round" counts from 0</div></div>`;
}
function bars(items, max, cut){
  let h = '<div class="bars">';
  items.forEach((it, i) => {
    if (cut != null && i === cut) h += `<div class="cut rv" data-at="${it.cutAt || 0}"><span>${esc(it.cutLabel || '')}</span></div>`;
    h += `<div class="b ${it.rowCls || ''}" data-cid="${it.cid}"><span>${esc(it.label)}</span><span class="tr"><span class="fl ${it.cls || ''}" style="width:0%" data-w="${(100*it.v/(max || 1)).toFixed(1)}"></span></span><span class="r" style="text-align:right">${esc(it.txt)}</span></div>`;
  });
  return h + '</div>';
}
function growBars(f){ insp.querySelectorAll('.bars .fl').forEach(e => { const w = (+e.dataset.w*f).toFixed(1) + '%'; if (e.style.width !== w) e.style.width = w; }); }
function spark(upto, mark){
  const s = (D.series || []).filter(x => x.f1 != null); if (s.length < 2) return '';
  const w = 460, h = 90, r0 = s[0].r, r1 = s[s.length - 1].r, lo = Math.min(...s.map(x => x.f1)), hi = Math.max(...s.map(x => x.f1));
  const X = r => 6 + (w - 12)*(r - r0)/((r1 - r0) || 1), Y = v => h - 14 - (h - 26)*(v - lo)/((hi - lo) || 1);
  const pts = s.filter(x => x.r <= upto).map(x => `${X(x.r).toFixed(1)},${Y(x.f1).toFixed(1)}`).join(' ');
  const all = s.map(x => `${X(x.r).toFixed(1)},${Y(x.f1).toFixed(1)}`).join(' ');
  const last = s.filter(x => x.r <= upto).slice(-1)[0];
  let g = `<svg class="spark" viewBox="0 0 ${w} ${h}">`;
  g += `<polyline points="${all}" fill="none" stroke="var(--line2)" stroke-width="1" stroke-dasharray="2 3"/>`;
  if (mark != null) g += `<line x1="${X(mark)}" x2="${X(mark)}" y1="6" y2="${h - 12}" stroke="var(--warn)" stroke-dasharray="3 3" stroke-width="1"/><text x="${X(mark) + 3}" y="14" font-size="9" font-family="var(--mono)" fill="var(--warn)">voting starts</text>`;
  if (pts) g += `<polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-linejoin="round"/>`;
  if (last) g += `<circle cx="${X(last.r)}" cy="${Y(last.f1)}" r="3" fill="var(--accent)"/><text x="${Math.min(w - 40, X(last.r) + 5)}" y="${Y(last.f1) - 5}" font-size="10" font-family="var(--mono)" fill="var(--ink)">${last.f1.toFixed(3)}</text>`;
  g += `<text x="6" y="${h - 2}" font-size="9" font-family="var(--mono)" fill="var(--muted)">round ${r0 + 1}</text><text x="${w - 6}" y="${h - 2}" font-size="9" font-family="var(--mono)" fill="var(--muted)" text-anchor="end">${r1 + 1}</text>`;
  return g + '</svg>';
}
function metricsHtml(val){
  const m = [['macro-F1', val.macro_f1], ['rare recall', val.rare_recall], ['missed attacks', val.missed_attack_rate]];
  return `<div class="metrics">` + m.map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v == null ? '–' : v.toFixed(3)}</div></div>`).join('') + `</div>`;
}
const ROLE = r => r ? `<span class="${r === 'poisoned' || r === 'rare' ? 'role-' + r : 'dim'}">${esc(r)}</span>` : '<span class="dim">–</span>';
// a drone's name coloured by its true role (only where the role is shown: owner's view, never the server's)
const nameRole = (c, r) => `<span class="${r === 'poisoned' || r === 'rare' ? 'role-' + r : ''}">${esc(droneName(c))}</span>`;
const roleOf = c => (sites.find(s => s.cid === c) || {}).role;

// ---------- scenes ----------
// Each scene: {step, key, title, dur, usesFocus, setup(), frame(t), cues:[[t, text]]}. frame(t) is a pure function of
// the scene clock, so pause, speed, seek and jumping between steps all work without bookkeeping.
const scenes = [];
const REAL = {};  // scene key -> the dashboard setting that runs it in a real training run (demo only)
function sceneEnroll(){
  const order = cids.slice(), arr = {};
  let pk = {}, keys = {};
  return {step: 1, key: 'enroll', title: 'Enrollment & key exchange', dur: 14,
    cues: [[0, 'Before take-off the operator’s certificate authority (CA) signs each drone’s public keys: a “certificate” binds drone id, Ed25519 signing key and X25519 key.'],
           [7.4, 'Each drone and the server run an X25519 key exchange; HKDF-SHA256 turns the shared secret into two AES-256-GCM session keys (uplink, downlink).'],
           [11, 'All links are encrypted and authenticated. The server accepts updates only from enrolled keys.']],
    setup(){
      base({link: 'off', ca: 1, st: 'not enrolled'}); hud(R0, 'Before round 1');
      order.forEach((c, i) => { arr[c] = 1 + i*0.55 + 0.9; pk[c] = packet('cert'); keys[c] = [packet('key'), packet('key')]; });
      let rows = '';
      order.forEach(c => rows += `<tr class="rv" data-at="${arr[c].toFixed(2)}"><td>${droneName(c)}</td><td class="dim">${esc(sites.find(s => s.cid === c).name)}</td><td>${esc(enrolled(c) || '—')}</td></tr>`);
      panel('Enrollment & key exchange', 'Done once per job, before the first round.',
        `<h4>Crypto suite</h4>${suite()}
         <div class="kv" style="margin-top:6px"><span class="k">Certificate</span><span>CA signature over (cid, Ed25519 key, X25519 key)</span>
         <span class="k">Fingerprint</span><span>SHA-256 of the certificate, first 16 hex</span>
         ${D.crypto.ca_fingerprint ? `<span class="k">Operator CA</span><span class="m">${esc(D.crypto.ca_fingerprint)}</span>` : ''}
         ${D.crypto.server_fingerprint ? `<span class="k">Server</span><span class="m">${esc(D.crypto.server_fingerprint)}</span>` : ''}
         ${D.crypto.enroll_ms != null ? `<span class="k">Enrollment time</span><span class="m">${fmtMs(D.crypto.enroll_ms)} (all drones)</span>` : ''}</div>
         <h4>Enrolled drones</h4><table class="t"><tr><th>Drone</th><th>Site</th><th>Fingerprint</th></tr>${rows}</table>
         ${rv(7.4, `<h4>Session keys</h4><div class="kv"><span class="k">Key exchange</span><span>X25519 (ECDH) per drone ↔ server</span><span class="k">Derivation</span><span>HKDF-SHA256 → uplink key, downlink key</span><span class="k">Packets</span><span>AES-256-GCM, 12-byte nonce, header as associated data</span></div>`)}
         ${rv(11, `<div class="note">Raw traffic never leaves a site. Only model updates travel, and only over these links.</div>`)}`);
    },
    frame(t){
      cids.forEach(c => {
        const a = arr[c], f = seg(t, a - 0.9, a);
        place(pk[c], CA, pos[c], f);
        const got = t >= a, sec = t >= 10.6;
        drone(c, {op: sec ? 1 : got ? .7 : .38, cert: got ? 1 : 0, st: sec ? 'secure link' : got ? 'certificate ✓' : 'not enrolled', cls: sec ? 'ok' : ''});
        const kf = seg(t, 7.6 + (c % 5)*0.12, 10.4);
        up(keys[c][0], c, kf); down(keys[c][1], c, kf);
        link(c, sec ? 'sec' : 'off', null, sec ? clamp((t - 10.6)/0.6) : 0);
      });
    }};
}
function trainTimes(R){
  const ms = cids.map(c => (R.L[c] && R.L[c].ms) || null), have = ms.filter(x => x != null);
  const mx = have.length ? Math.max(...have) : 1;
  const fin = {}; cids.forEach((c, i) => fin[c] = 0.8 + 10.4*((ms[i] != null ? ms[i] : mx*(0.55 + 0.045*i))/mx));
  return {fin, real: have.length > 0, mx};
}
function sceneTrain(R){
  let tt, cells = {};
  return {step: 2, key: 'train', title: 'Local training at the 10 sites', dur: 14,
    cues: [[0, `Round ${rnum(R)}: every site trains the shared model on its own labelled traffic. Raw traffic never leaves the site.`],
           [4, 'Faded with a ring = still training. Full colour = connected and finished: ready to send its update.'],
           [12, 'All updates are ready.']],
    setup(){
      base(); hud(R); tt = trainTimes(R);
      let rows = '';
      sites.forEach(s => {
        const l = R.L[s.cid] || {};
        rows += `<tr data-cid="${s.cid}"><td>${nameRole(s.cid, s.role)}</td><td class="r">${fmtInt(s.n)}</td><td class="r">${l.steps ?? '–'}</td><td class="r">${l.ms != null ? (l.ms/1000).toFixed(2) + ' s' : '–'}</td><td class="r">${l.norm != null ? l.norm.toFixed(3) : '–'}</td><td class="st">training…</td><td>${ROLE(s.role)}</td></tr>`;
      });
      panel(`Round ${rnum(R)} · local training`, tt.real ? 'Real local training times from the run; the animation keeps their ratios.' : 'Local timings not in this log; order is illustrative.',
        `<table class="t"><tr><th>Drone</th><th class="r">train rows</th><th class="r">steps</th><th class="r">time</th><th class="r">‖Δ‖</th><th>state</th><th>true role*</th></tr>${rows}</table>
         <div class="note">* known to the site owner only; the server never sees roles or raw traffic.</div>
         <div class="note">‖Δ‖ = size of the model update this site will send.</div>`);
      insp.querySelectorAll('tr[data-cid]').forEach(tr => cells[tr.dataset.cid] = tr);
    },
    frame(t){
      cids.forEach(c => {
        const f = tt.fin[c], done = t >= f, l = R.L[c] || {};
        drone(c, {op: done ? 1 : .42, ring: done ? null : t/f, spin: done ? 0 : t*540,
          st: done ? (l.ms != null ? `ready · ${(l.ms/1000).toFixed(2)} s` : 'ready') : `training… ${Math.floor(100*t/f)}%`, cls: done ? 'ok' : ''});
        const tr = cells[c]; if (tr){ const td = tr.querySelector('.st'), s = done ? 'ready' : 'training…'; if (td.textContent !== s){ td.textContent = s; td.className = 'st ' + (done ? 'ok' : 'dim'); } }
      });
    }};
}
function sceneEncrypt(R){
  let F, u, pk, others = {};
  return {step: 3, key: 'encrypt', title: 'Encryption', dur: 16, usesFocus: true,
    cues: [[0, 'The update is a list of float32 numbers. These are the first 8 real values from the focused drone (click another drone to switch).'],
           [3, 'They are written as little-endian bytes: 4 bytes per value.'],
           [6, 'The drone signs header + bytes with its Ed25519 key, so the server can prove who sent them and for which round.'],
           [9, 'AES-256-GCM encrypts them with the uplink session key. The 16-byte tag will expose any change in transit.'],
           [13, 'Every drone does the same; only ciphertext leaves the drone.']],
    setup(){
      base(); hud(R); F = defaultFocus(R); u = R.U[F];
      cids.forEach(c => DN[c].g.classList.toggle('focus', c === F));
      pk = packet('up'); cids.forEach(c => { if (c !== F) others[c] = packet('up'); });
      if (!u){ panel('Encryption', '', '<div class="note">No uplink record for this drone.</div>'); return; }
      const pw = plainWords(u);
      panel(`Encryption at ${droneName(F)}`, `${esc(sites.find(s => s.cid === F).name)} · round ${rnum(R)} · real values`,
        `<h4>1 · Local update (first 8 of the float32 values)</h4>${valsGrid(u.ps)}
         ${rv(3, `<h4>2 · Plaintext bytes (float32, little-endian)</h4>` + (pw.aligned ? valsGrid(u.ps, pw.w) : `<div class="hex">${esc(words(u.ph, 12))}</div>`))}
         ${rv(6, `<h4>3 · Ed25519 signature over header + bytes · 64 B</h4><div class="hex dim">${esc(shortHex(u.sig, 48, 16))}</div>`)}
         ${rv(9, `<h4>4 · AES-256-GCM packet</h4><div class="anat"><span class="a" style="flex:1">hdr</span><span class="n" style="flex:1.1">nonce</span><span class="c" style="flex:7">ciphertext</span><span class="g" style="flex:1.3">tag</span><span class="s" style="flex:2">sig</span></div><div class="note" style="margin:0 0 5px">layout schematic, not to scale</div>${packetBlock(u, R)}`)}
         ${rv(13, `<div class="kv" style="margin-top:8px"><span class="k">Packet size</span><span class="m">${fmtInt(u.bytes)} B</span><span class="k">Sign + encrypt</span><span class="m">${fmtMs(u.enc)}</span><span class="k">Session key</span><span>uplink key of ${droneName(F)} (from step 1)</span></div>`)}`);
    },
    frame(t){
      cids.forEach(c => drone(c, {op: 1, st: c === F ? (t < 9 ? 'preparing update' : t < 12 ? 'encrypting…' : 'encrypted ✓') : (t >= 13.5 ? 'encrypted ✓' : 'ready'), cls: (c === F && t >= 12) || (c !== F && t >= 13.5) ? 'ok' : (c === F ? 'acc' : '')}));
      const L = links[F];
      if (t >= 9.6) show(pk, L.A, L.ang); else hide(pk);
      pk.setAttribute('opacity', seg(t, 9.6, 10.6));
      cids.forEach(c => { if (c !== F){ if (t >= 13.4 + (c % 5)*0.1) show(others[c], links[c].A, links[c].ang); else hide(others[c]); } });
    }};
}
function sendPlan(R){
  const tt = trainTimes(R), order = cids.slice().sort((a, b) => tt.fin[a] - tt.fin[b]);
  const tx = cids.map(c => (R.U[c] && R.U[c].tx) || null).filter(x => x != null), mx = tx.length ? Math.max(...tx) : 1;
  const plan = {}; order.forEach((c, i) => { const x = (R.U[c] && R.U[c].tx) || mx; plan[c] = {dep: 0.6 + i*0.45, dur: 2.4*(0.75 + 0.25*x/mx)}; plan[c].arr = plan[c].dep + plan[c].dur; });
  return plan;
}
// One drone's packet on its link in one direction, with the re-sends logged for it this round (R.events): legs
// {k: 'pkt' | 'nack', a, b, att, ev} in scene seconds. A rejected packet is followed by a NACK the other way, a short
// pause (re-sealing with a fresh nonce) and the next attempt.
function legs(R, c, dir, dep, dur){
  const evs = (R.events || []).filter(e => e.cid === c && e.dir === dir).sort((a, b) => a.attempt - b.attempt);
  const out = []; let t0 = dep;
  evs.forEach((e, i) => { out.push({k: 'pkt', a: t0, b: t0 + dur, att: i + 1, ev: e}); out.push({k: 'nack', a: t0 + dur + 0.35, b: t0 + dur + 1.75, ev: e}); t0 += dur + 2.3; });
  out.push({k: 'pkt', a: t0, b: t0 + dur, att: evs.length + 1});
  return out;
}
const legsEnd = L => L[L.length - 1].b;
// Draw one drone's legs at time t; -> {ph: wait | fly | rejected | nack | reseal | done, att, ev}
function runLegs(c, L, t, pk, nk, fromDrone, kind){
  const mv = fromDrone ? up : down, back = fromDrone ? down : up, lk = links[c];
  let i = 0; while (i < L.length && t >= L[i].b) i++;
  if (i === L.length){ hide(pk); hide(nk); return {ph: 'done', att: L.length > 1 ? L[L.length - 1].att : 1}; }
  const l = L[i];
  if (t < l.a){
    hide(nk);
    if (i === 0){ setA(cache, pk, 'class', 'pkt ' + kind); show(pk, fromDrone ? lk.A : lk.B, lk.ang); return {ph: 'wait'}; }
    const prev = L[i - 1];
    if (prev.k === 'pkt'){ setA(cache, pk, 'class', 'pkt bad'); show(pk, fromDrone ? lk.B : lk.A, lk.ang); return {ph: 'rejected', ev: prev.ev}; }
    hide(pk); return {ph: 'reseal', att: l.att};
  }
  const f = (t - l.a)/(l.b - l.a);
  if (l.k === 'nack'){ hide(pk); back(nk, c, f); return {ph: 'nack', ev: l.ev}; }
  hide(nk); setA(cache, pk, 'class', 'pkt ' + (l.att > 1 ? 'resend' : kind)); mv(pk, c, f);
  return {ph: 'fly', att: l.att};
}
function legState(r, sent, sending){
  return r.ph === 'done' ? [r.att > 1 ? `${sent} (try ${r.att})` : sent, 'ok'] : r.ph === 'fly' ? [r.att > 1 ? `re-sending… try ${r.att}` : sending, r.att > 1 ? 'nack' : 'acc']
    : r.ph === 'rejected' ? [`rejected · ${r.ev.error}`, 'bad'] : r.ph === 'nack' ? ['NACK: send again', 'nack'] : r.ph === 'reseal' ? ['re-encrypting (new nonce)', 'nack'] : null;
}
function evNote(R, dirs){
  const ev = (R.events || []).filter(e => dirs.includes(e.dir));
  if (!ev.length) return '';
  return `<div class="note"><span class="badge nack">⟲ ${ev.length} re-send${ev.length > 1 ? 's' : ''}</span> ` + ev.map(e => `${esc(droneName(e.cid))} ${esc(e.dir)} try ${e.attempt}: ${esc(e.error)} → NACK ${fmtInt(e.nack)} B → re-sent ${fmtInt(e.resend)} B`).join(' · ') + '</div>';
}
function sceneTransmit(R){
  let pk = {}, nk = {}, cells = {};
  const plan = sendPlan(R), LG = {}; cids.forEach(c => LG[c] = legs(R, c, 'uplink', plan[c].dep, plan[c].dur));
  const end = Math.max(...cids.map(c => legsEnd(LG[c])));
  return {step: 4, key: 'transmit', title: 'Transmission over the links', dur: Math.max(12, end + 2.5),
    cues: [[0, 'Drones send as soon as they are ready. Each packet carries only ciphertext, nonce, tag and signature.'],
           [5, 'A check at the server marks each packet that has arrived. Nothing is decrypted yet.'],
           [9.5, (R.events || []).some(e => e.dir === 'uplink') ? 'A rejected packet (amber link) is answered with a NACK and sent again with a fresh nonce: this round logged re-sends.' : 'Transfer times are computed from the packet size and the simulated 10 Mbit/s link model of the run.']],
    setup(){
      base(); hud(R);
      cids.forEach(c => { pk[c] = packet('up', `${droneName(c)} → server · ${fmtInt(R.U[c] && R.U[c].bytes)} B`); nk[c] = packet('nack', 're-send request (NACK)'); });
      let rows = '', tot = 0;
      cids.forEach(c => { const u = R.U[c] || {}; tot += u.bytes || 0;
        rows += `<tr data-cid="${c}"><td>${droneName(c)}</td><td class="r">${fmtInt(u.bytes)}</td><td class="r">${fmtS(u.tx)}</td><td class="r">${fmtMs(u.enc)}</td><td class="r">${u.att || 1}</td><td class="st dim">waiting</td></tr>`; });
      panel(`Round ${rnum(R)} · uplink`, 'Encrypted model updates (signed by the drone process) on their way to the aggregation server.',
        `<table class="t"><tr><th>Drone</th><th class="r">bytes</th><th class="r">transfer*</th><th class="r">crypto</th><th class="r">tries</th><th>state</th></tr>${rows}</table>
         <div class="kv" style="margin-top:8px"><span class="k">Uplink total</span><span class="m">${fmtB(tot)}</span></div>
         ${evNote(R, ['uplink'])}
         <div class="note">* simulated transfer time logged by the run: packet size over ${D.crypto.bandwidth_mbps ? D.crypto.bandwidth_mbps + ' Mbit/s' : 'the configured bandwidth'} plus ${D.crypto.latency_ms != null ? D.crypto.latency_ms + ' ms' : ''} latency.</div>`);
      insp.querySelectorAll('tr[data-cid]').forEach(tr => cells[tr.dataset.cid] = tr);
    },
    frame(t){
      cids.forEach(c => {
        const r = runLegs(c, LG[c], t, pk[c], nk[c], true, 'up'), arrived = r.ph === 'done', redo = LG[c].length > 1 && t >= LG[c][0].b && !arrived;
        slot(c, arrived ? 'recv' : r.ph === 'rejected' ? 'bad' : redo ? 'nack' : '');
        link(c, redo ? 'resend' : 'sec');
        const [st, cls] = legState(r, 'sent ✓', 'sending…') || ['ready', ''];
        drone(c, {op: 1, st, cls});
        const td = cells[c] && cells[c].querySelector('.st'), s = arrived ? 'received' : redo ? 're-send' : r.ph === 'fly' ? 'in flight' : 'waiting';
        if (td && td.textContent !== s){ td.textContent = s; td.className = 'st ' + (arrived ? 'ok' : redo ? 'nackc' : r.ph === 'fly' ? '' : 'dim'); }
      });
    }};
}
function sceneVerify(R){
  let F, u, cells = {};
  const at = {}; cids.forEach((c, i) => at[c] = 1 + i*0.75);
  return {step: 5, key: 'verify', title: 'Server: tag check, decrypt, verify', dur: 14, usesFocus: true,
    cues: [[0, 'The server checks the GCM tag first. A wrong key or any changed byte fails here, before anything is decrypted.'],
           [3.2, 'Tag OK: decrypt with that drone’s uplink session key.'],
           [6.2, 'Then the Ed25519 signature (over header + update bytes) is checked against the key the CA certified for that drone.'],
           [9.2, 'The decrypted values are exactly the values the drone sent: bit-identical.']],
    setup(){
      base(); hud(R); F = defaultFocus(R); u = R.U[F];
      cids.forEach(c => DN[c].g.classList.toggle('focus', c === F));
      if (!u){ panel('Server checks', '', '<div class="note">No uplink record.</div>'); return; }
      const same = u.ok && u.ps.length > 0 && u.ps.length === u.ds.length && u.ps.every((x, i) => x === u.ds[i]);
      const pw = plainWords(u);
      const cmp = u.ps.map((x, i) => `<tr><td class="dim">${i}</td><td>${fmtF(x)}</td><td>${fmtF(u.ds[i])}</td>${pw.aligned ? `<td class="m">${pw.w[i]}</td>` : ''}<td class="${x === u.ds[i] ? 'ok' : 'bad'}">${x === u.ds[i] ? '=' : '≠'}</td></tr>`).join('');
      let rows = '';
      cids.forEach(c => { const v = R.U[c] || {}; rows += `<tr class="rv" data-at="${at[c]}"><td>${droneName(c)}</td><td class="${v.ok ? 'ok' : 'bad'}">${v.ok ? 'ok' : esc(v.err || 'fail')}</td><td class="${v.ok ? 'ok' : 'bad'}">${v.ok ? 'ok' : '—'}</td><td class="r">${fmtMs(v.dec)}</td></tr>`; });
      panel(`Server checks · ${droneName(F)}`, `Round ${rnum(R)} · what the aggregation server does with each packet`,
        `<h4>Received packet</h4>${packetBlock(u, R)}
         <div class="step rv done" data-at="1.5"><span class="d">1</span><div><b>GCM tag check</b> <span class="ok">✓ valid</span><div class="note" style="margin:0">header + ciphertext unchanged since encryption</div></div></div>
         <div class="step rv done" data-at="3.4"><span class="d">2</span><div><b>Decrypt</b> with the uplink session key → ${fmtInt(u.bytes)} B packet, ${fmtMs(u.dec)}</div></div>
         <div class="step rv ${u.ok ? 'done' : 'fail'}" data-at="6.4"><span class="d">3</span><div><b>Ed25519 signature</b> ${u.ok ? '<span class="ok">✓ verified</span>' : '<span class="bad">✗ failed</span>'}<div class="note" style="margin:0">against the certified key, fingerprint <span class="m">${esc(enrolled(F) || '—')}</span></div></div></div>
         ${rv(9.2, `<h4>Sent by the drone vs decrypted at the server</h4><table class="t cmp"><tr><th>#</th><th>sent</th><th>decrypted</th>${pw.aligned ? '<th>bytes</th>' : ''}<th></th></tr>${cmp}</table>
           <div style="margin-top:8px">${same ? '<span class="badge ok">bit-identical ✓</span>' : '<span class="badge bad">mismatch</span>'}</div>`)}
         <h4>All drones this round</h4><table class="t"><tr><th>Drone</th><th>tag</th><th>signature</th><th class="r">decrypt</th></tr>${rows}</table>`);
    },
    frame(t){
      srv.classList.toggle('flash', t < 13 && Math.floor(t*2) % 2 === 0);
      cids.forEach(c => {
        const v = R.U[c] || {}, done = t >= at[c];
        slot(c, done ? (v.ok === false ? 'bad' : 'ok') : 'recv');
        drone(c, {op: 1, st: done ? (v.ok === false ? 'rejected' : 'verified ✓') : 'sent ✓', cls: done && v.ok === false ? 'bad' : 'ok'});
      });
      const pk = insp.querySelector('.pk'); if (pk) pk.classList.toggle('hot', t >= 0.4 && t < 1.5);
    }};
}
function sceneFilter(R){
  const K = R.sel.length;
  return {step: 6, key: 'filter', title: 'Filter (closest updates)', dur: 12,
    cues: [[0, 'Jiang’s filter: for each update the server adds up its squared distances to all the other updates.'],
           [4.5, `The ${K} updates with the smallest sums (the most typical ones) are kept.`],
           [8, vs != null && R.round < vs ? `The others are held back. Voting starts at round ${vs + 1}; until then they are simply left out.` : 'The others are held back for the vote.']],
    setup(){
      base(); hud(R);
      cids.forEach(c => slot(c, 'ok'));
      const items = R.rows.filter(x => x.score != null).sort((a, b) => a.score - b.score).map((x, i) => ({cid: x.cid, label: droneName(x.cid), v: x.score,
        txt: x.score.toFixed(2), cls: R.sel.includes(x.cid) ? 'k' : 'h', cutAt: 4.5, cutLabel: `${K} closest kept`}));
      const mx = items.length ? items[items.length - 1].v : 1;
      panel(`Round ${rnum(R)} · filter`, 'Score = Σ squared distance to the other updates (smaller = more typical).',
        items.length ? bars(items, mx, K) + `<div class="note">The server sees only the updates. It does not know which drone is honest, rare or poisoned.</div>`
                     : '<div class="note">No server rows for this round.</div>');
    },
    frame(t){
      growBars(ease(seg(t, .4, 3.8)));
      insp.querySelectorAll('[data-at]').forEach(e => e.classList.toggle('on', t >= +e.dataset.at));
      const decided = t >= 4.5;
      cids.forEach(c => {
        const kept = R.sel.includes(c);
        link(c, decided ? (kept ? 'sel' : 'held') : 'sec', decided && kept ? 1.8 : 1.2);
        slot(c, decided ? (kept ? 'sel' : 'held') : 'ok');
        drone(c, {op: 1, st: decided ? (kept ? 'kept' : 'held back') : 'scoring…', cls: decided ? (kept ? 'acc' : 'warn') : ''});
      });
    }};
}
function sceneFF(from, to, title){
  let ups = {}, downs = {};
  const n = Math.max(1, to.round - from.round - 1), dur = n > 2 ? 8 : 5;
  return {step: null, key: 'ff', title, dur,
    cues: [[0, `Rounds ${from.round + 2}–${to.round} repeat the same cycle: train, encrypt, send, verify, filter, aggregate, broadcast.`],
           [dur*0.6, vs != null && to.round >= vs && from.round < vs ? `Voting starts at round ${vs + 1}.` : 'Fast-forward.']],
    setup(){
      base(); cids.forEach(c => { ups[c] = packet('up'); downs[c] = packet('down'); });
      const s = (D.series || []).filter(x => x.r > from.round && x.r < to.round);
      const bytes = s.reduce((a, x) => a + (x.bytes || 0), 0), ms = s.reduce((a, x) => a + (x.ms || 0), 0), rt = s.reduce((a, x) => a + (x.rt || 0), 0);
      panel('Fast-forward', `Rounds ${from.round + 2}–${to.round}`,
        `<h4>Validation macro-F1 of the global model</h4><div id="spk"></div>
         <div class="kv" style="margin-top:8px"><span class="k">Link traffic in these rounds</span><span class="m">${fmtB(bytes)}</span><span class="k">Crypto time</span><span class="m">${(ms/1000).toFixed(2)} s</span>${rt ? `<span class="k">Packets re-sent after an interception</span><span class="m nackc">${rt}</span>` : ''}</div>`);
      this._last = null;
    },
    frame(t){
      const f = t/dur, r = Math.min(to.round - 1, from.round + 1 + Math.floor(f*n));
      hud({round: r}); srvSub.textContent = '';
      if (this._last !== r){ this._last = r; const e = $('#spk'); if (e) e.innerHTML = spark(r, vs); }
      const cyc = dur/n, ph = (t % cyc)/cyc;
      cids.forEach(c => { up(ups[c], c, ph/0.45); down(downs[c], c, (ph - 0.55)/0.45); slot(c, ph > 0.45 && ph < 0.6 ? 'ok' : '');
        drone(c, {op: ph < 0.3 ? .55 : 1, st: ''}); });
    }};
}
function sceneVote(R){
  const voted = R.voted, N0 = (voted[0] && voted[0].total) || N;
  let cp = {}, bl = {};
  return {step: 7, key: 'vote', title: 'Vote', dur: 16,
    cues: [[0, `Round ${rnum(R)}: voting is active. The held-back updates are not thrown away yet.`],
           [0.8, 'The server sends the filtered model and a copy of each held-back model to the voters, over each drone’s own encrypted vote link.'],
           [4, 'Each drone tests the copies on its own local data and votes yes if a model is about as good as the filtered model.'],
           [7.2, 'Each voter’s scores come back to the server over its encrypted ballot link.'],
           [10, `An update is voted back in with more than half of the ${N0} votes.`]],
    setup(){
      base(); hud(R);
      cids.forEach(c => { cp[c] = packet('vote', `copies of held-back models → ${droneName(c)}`); bl[c] = packet('ballot'); });
      let items = voted.slice().sort((a, b) => b.yes - a.yes).map(x => ({cid: x.cid, label: droneName(x.cid), v: x.yes, txt: `${x.yes}/${x.total || N0}`,
        cls: x.yes > (x.total || N0)/2 ? 'okb' : 'badb'}));
      const rows = voted.map(x => { const inn = x.yes > (x.total || N0)/2; const st = x.status === 'vetoed' ? 'voted in, weight 0' : inn ? 'voted back in' : 'voted out';
        return `<tr><td>${nameRole(x.cid, x.role)}</td><td class="r">${x.yes}/${x.total || N0}</td><td class="${inn ? 'ok' : 'bad'}">${st}</td><td>${ROLE(x.role)}</td></tr>`; }).join('');
      panel(`Round ${rnum(R)} · vote`, `Held back by the filter: ${R.drop.map(droneName).join(', ') || 'none'}`,
        voted.length ? `<h4>Yes votes (threshold: more than ${N0/2})</h4>${bars(items, N0)}
          ${rv(12.5, `<h4>Outcome</h4><table class="t"><tr><th>Drone</th><th class="r">yes</th><th>verdict</th><th>true role*</th></tr>${rows}</table>
          <div class="note">* not visible to the server; shown so you can judge the vote. The tallies are logged; who voted what is not.</div>`)}
          ${rv(1, `<div class="kv" style="margin-top:8px"><span class="k">Vote copies</span><span class="m">${fmtInt(R.vote.messages)} messages · ${fmtB(R.vote.bytes)}</span>${R.vote.ballots != null ? `<span class="k">Ballots</span><span class="m">${fmtInt(R.vote.ballots)} messages · ${fmtB(R.vote.ballot_bytes)}</span>` : ''}</div>${evNote(R, ['vote', 'ballot'])}`)}`
        : '<div class="note">No vote was logged in this round.</div>');
    },
    frame(t){
      growBars(ease(seg(t, 10, 12.5)));
      const vd = t >= 12.5;
      cids.forEach((c, i) => {
        const v = R.W[c], isV = v && v.yes != null, inn = isV && v.yes > (v.total || N0)/2;
        down(cp[c], c, seg(t, 0.8 + i*0.12, 3.6 + i*0.12));
        up(bl[c], c, seg(t, 7.2 + i*0.1, 9.6 + i*0.1));
        const evalf = seg(t, 4, 7);
        link(c, isV ? (vd ? (inn ? 'back' : 'rej') : 'held') : 'sel', isV && vd && inn ? 1.8 : (!isV ? 1.6 : 1.2));
        slot(c, isV ? (vd ? (inn ? 'ok' : 'bad') : 'held') : 'sel');
        drone(c, {op: isV && vd && !inn ? .6 : 1, ring: evalf > 0 && evalf < 1 ? evalf : null,
          st: vd && isV ? (inn ? 'voted back in' : 'voted out') : evalf > 0 && evalf < 1 ? 'testing copies…' : t >= 7.2 && t < 10 ? 'vote sent' : isV ? 'held back' : 'kept',
          cls: vd && isV ? (inn ? 'ok' : 'bad') : isV ? 'warn' : 'acc'});
      });
    }};
}
function sceneDQN(R){
  const K = R.sel.length, fixed = 1/(K + 1);
  const contrib = R.rows.filter(x => x.w > 0), mx = Math.max(1e-9, ...R.rows.map(x => x.w));
  return {step: 8, key: 'dqn', title: 'DQN weight', dur: 12,
    cues: [[0, 'For each update voted back in, the DQN agent chooses how much it should count.'],
           [4, `The fixed-weight variant would give each one 1/(K+1) = ${fixed.toFixed(3)}. The DQN learns from the reward on the server’s public check data.`],
           [8, 'Line thickness = share in the new global model. Voted-out updates count zero.']],
    setup(){
      base(); hud(R);
      const re = R.readm.map(c => ({c, w: R.dqn[String(c)] ?? (R.W[c] && R.W[c].dqn)}));
      const reHtml = re.length ? re.map(x => `<tr><td>${droneName(x.c)}</td><td class="r">${x.w != null ? x.w.toFixed(4) : '–'}</td><td class="r dim">${fixed.toFixed(4)}</td><td class="r">${x.w != null ? (x.w/fixed).toFixed(2) + '×' : ''}</td></tr>`).join('')
        : '';
      const items = R.rows.slice().sort((a, b) => b.w - a.w).map(x => ({cid: x.cid, label: droneName(x.cid), v: x.w, txt: (100*x.w).toFixed(1) + ' %',
        cls: R.readm.includes(x.cid) ? 'okb' : R.sel.includes(x.cid) ? 'k' : ''}));
      panel(`Round ${rnum(R)} · DQN weight`, 'How much each update counts in this round’s global model.',
        (re.length ? `<h4>DQN choice for the updates voted back in</h4><table class="t"><tr><th>Drone</th><th class="r">DQN w</th><th class="r">fixed w</th><th class="r">ratio</th></tr>${reHtml}</table>`
                   : '<div class="note">No update was voted back in this round, so the DQN had nothing to weigh.</div>') +
        rv(3, `<h4>Share in the new global model</h4>${bars(items, mx)}<div class="note">blue = kept by the filter (by data size) · green = voted back in (DQN weight) · grey = left out</div>`));
    },
    frame(t){
      growBars(ease(seg(t, 3.2, 7.5)));
      const g = ease(seg(t, 4, 8));
      cids.forEach(c => {
        const x = R.W[c] || {w: 0}, on = x.w > 0, back = R.readm.includes(c);
        link(c, on ? (back ? 'back' : 'sel') : (R.voted.find(v => v.cid === c) ? 'rej' : 'held faint'), on ? 1.2 + g*7*x.w/mx : 1);
        slot(c, on ? (back ? 'ok' : 'sel') : 'held');
        drone(c, {op: on ? 1 : .55, st: on ? `share ${(100*x.w).toFixed(1)} %` : 'not counted', cls: on ? (back ? 'ok' : 'acc') : 'dim'});
      });
    }};
}
function sceneAggregate(R){
  const mx = Math.max(1e-9, ...R.rows.map(x => x.w));
  let dots = {}, dl = {}, nk = {};
  const plan = {}, LG = {}, txm = Math.max(1e-9, ...R.down.map(d => d.tx || 1));
  cids.forEach((c, i) => { plan[c] = {dep: 4.4 + i*0.18, dur: 2.6 * (0.75 + 0.25*((R.Dn[c] && R.Dn[c].tx) || 1)/txm)};
    LG[c] = legs(R, c, 'downlink', plan[c].dep, plan[c].dur); });
  const nxt = Math.max(11.5, ...cids.map(c => legsEnd(LG[c]) + 0.6)), dur = nxt + 2.5;
  return {step: 9, key: 'aggregate', title: 'Aggregate & broadcast', dur,
    cues: [[0, 'The server merges the updates with those weights: global = (M_B + Σ wᵢ·Mᵢ) / (1 + Σ wᵢ), where M_B is the data-size-weighted mean of the kept updates.'],
           [4.4, 'The new global model goes back to every drone over its encrypted downlink, including drones whose update was left out.'],
           [nxt, 'Next round: every drone starts training from the new global model.']],
    setup(){
      base(); hud(R);
      cids.forEach(c => { dots[c] = S('circle', {r: 3, class: 'dot', style: 'display:none'}, gPk); dl[c] = packet('down', `global model → ${droneName(c)}`); nk[c] = packet('nack', `${droneName(c)} asks for the global model again (NACK)`); });
      let rows = '', tot = 0;
      cids.forEach(c => { const d = R.Dn[c] || {}; tot += d.bytes || 0; rows += `<tr><td>${droneName(c)}</td><td class="r">${fmtInt(d.bytes)}</td><td class="r">${fmtS(d.tx)}</td><td class="r">${d.att || 1}</td></tr>`; });
      panel(`Round ${rnum(R)} · aggregate & broadcast`, 'New global model, then the downlink.',
        `<h4>New global model · first 8 values</h4>${R.g.length ? valsGrid(R.g) : '<div class="note">not logged</div>'}
         ${rv(1.8, `<h4>Global model on the validation split</h4>${metricsHtml(R.val)}`)}
         ${rv(4.4, `<h4>Downlink (encrypted, per-drone key)</h4><table class="t"><tr><th>Drone</th><th class="r">bytes</th><th class="r">transfer</th><th class="r">tries</th></tr>${rows}</table><div class="kv" style="margin-top:6px"><span class="k">Downlink total</span><span class="m">${fmtB(tot)}</span></div>${evNote(R, ['downlink'])}`)}`);
    },
    frame(t){
      srv.classList.toggle('flash', t > 2.6 && t < 4.2);
      srvSub.textContent = t >= 3.4 ? `global model · round ${rnum(R)}` : '';
      cids.forEach(c => {
        const x = R.W[c] || {w: 0}, on = x.w > 0, d = dots[c];
        const f = seg(t, 0.4, 3.0);
        if (on && f > 0 && f < 1){ const p = lerp(links[c].S, SV, ease(f)); d.style.display = ''; d.setAttribute('cx', p.x); d.setAttribute('cy', p.y); d.setAttribute('r', 1.8 + 4*x.w/mx); }
        else d.style.display = 'none';
        const p = plan[c], r = runLegs(c, LG[c], t, dl[c], nk[c], false, 'down'), got = r.ph === 'done';
        const redo = LG[c].length > 1 && t >= LG[c][0].b && !got;
        link(c, redo ? 'resend' : t < 3.4 ? (on ? 'sel' : 'sec faint') : 'sec', t < 3.4 && on ? 1.2 + 5*x.w/mx : redo ? 1.8 : 1.2);
        slot(c, t < 3.4 ? (on ? 'sel' : '') : '');
        const next = t >= nxt + (c % 5)*0.15;
        const ls = !next && t >= p.dep && !got ? legState(r, '', 'receiving…') : null;
        drone(c, {op: next ? .42 : 1, ring: next ? seg(t, nxt, nxt + 4.5) : null, spin: next ? t*540 : 0,
          st: next ? 'training…' : got ? `model r${rnum(R)} ✓` + (r.att > 1 ? ` (try ${r.att})` : '') : ls ? (r.ph === 'rejected' ? 'tag ✗ · drone drops it' : r.ph === 'nack' ? 'NACK: send again' : ls[0]) : '',
          cls: got && !next ? 'ok' : ls ? ls[1] : ''});
      });
    }};
}
function sceneTamper(R, T){
  const c0 = T.cid, before = T.before || '', after = T.after || '', NK = T.nack, RS = T.resend, err = T.error || 'InvalidTag';
  const hasRe = !!(NK && RS && RS.nonce);
  let diff = -1; for (let i = 0; i < Math.min(before.length, after.length); i += 2) if (before.substr(i, 2) !== after.substr(i, 2)){ diff = i; break; }
  const fb = T.byte != null ? T.byte : (diff >= 0 ? diff/2 : null);
  const L = links[c0], mid = lerp(L.A, L.B, .5), off = {x: mid.x + L.uy*34, y: mid.y - L.ux*34};
  let pk = {}, att, nk, rs, dl = {};
  const hexMark = (h, i) => i < 0 ? esc(words(h, 12)) : (() => { const ws = words(h, 12), j = i + Math.floor(i/8); return esc(ws.slice(0, j)) + `<span class="flipped">${esc(ws.substr(j, 2))}</span>` + esc(ws.slice(j + 2)); })();
  const bits = () => { let x = parseInt(before.substr(diff, 2), 16) ^ parseInt(after.substr(diff, 2), 16), n = 0; while (x){ n += x & 1; x >>= 1; } return n === 1 ? '1 bit' : n + ' bits'; };
  // scene clock (s): interception 0-7.6 · rejection 7.6 · NACK 9-11.4 · re-seal 11.6-13.4 · re-send 13.6-16.4 · accepted 16.6 · round continues 19-24.5
  const X = {nack: [9, 11.4], seal: [11.6, 13.4], fly: [13.6, 16.4], ok: 16.6, cont: 19, down: [20.6, 23.2]};
  const st0 = (R.W[c0] || {}).status, kept = R.sel.includes(c0);
  return {step: 10, key: 'tamper', title: hasRe ? 'Interception → re-send' : 'Tamper attempt rejected', dur: hasRe ? 24.5 : 14,
    cues: hasRe ? [
      [0, `Round ${rnum(R)}: ${droneName(c0)} sends its update. On its link a simulated attacker alters the packet.`],
      [3.6, `It flips ${diff >= 0 ? bits() : 'one bit'} of ciphertext byte ${fb ?? '?'} and forwards the packet.`],
      [7.6, `GCM tag check fails (${err}): the server drops the packet before decrypting anything.`],
      [X.nack[0], `The server asks for the update again: a ${fmtInt(NK.bytes)} B NACK, itself signed and encrypted, goes back to ${droneName(c0)}.`],
      [X.seal[0], 'The drone re-encrypts the SAME update with a fresh random nonce, so the new ciphertext looks completely different.'],
      [X.fly[0], 'The new packet crosses the link; this time nothing touches it.'],
      [X.ok, 'Tag ✓ · decrypt ✓ · signature ✓: the decrypted values are bit-identical to what the drone sent.'],
      [X.cont, `The round carries on with all ${N} updates: filter, vote, weights, aggregate, then the new global model goes out.`]]
    : [[0, `Round ${rnum(R)}: an attacker on ${droneName(c0)}’s link intercepts the packet.`],
       [3.6, 'It flips a single byte of the ciphertext and forwards the packet.'],
       [7.6, `The GCM tag no longer matches: ${err}. The server rejects the packet before decrypting it.`],
       [10, 'Nothing from the forged packet reaches aggregation. (This trace has no re-send record.)']],
    setup(){
      base(); hud(R);
      cids.forEach(c => { pk[c] = packet('up'); dl[c] = packet('down'); });
      nk = packet('nack', 're-send request (NACK), server → drone'); rs = packet('resend', 're-sent update, fresh nonce');
      att = S('g', {class: 'mitm', transform: `translate(${off.x},${off.y})`, opacity: 0}, top);
      S('circle', {r: 10}, att); S('line', {x1: -6, y1: 0, x2: 6, y2: 0}, att); S('line', {x1: 0, y1: -6, x2: 0, y2: 6}, att);
      S('text', {y: -15}, att).textContent = 'interception (simulated attacker)';
      this.beam = S('line', {x1: off.x, y1: off.y, x2: mid.x, y2: mid.y, class: 'beam', opacity: 0}, top);
      const al = S('g', {class: 'alarmtxt', transform: `translate(${SV.x},${SV.y - 60})`, opacity: 0}, top);
      S('rect', {x: -84, y: -10, width: 168, height: 19, rx: 4}, al); S('text', {y: 4}, al).textContent = `${err} · rejected`;
      this.al = al;
      const src = T.source === 'run' ? 'real record from the run' : 'separate throwaway enrollment';
      const cmp = hasRe && RS.ps.length ? RS.ps.map((x, i) => `<tr><td class="dim">${i}</td><td>${fmtF(x)}</td><td>${fmtF(RS.ds[i])}</td><td class="${x === RS.ds[i] ? 'ok' : 'bad'}">${x === RS.ds[i] ? '=' : '≠'}</td></tr>`).join('') : '';
      const nackBody = NK && NK.body ? JSON.stringify(NK.body) : '';
      panel(`${hasRe ? 'Interception → re-send' : 'Tamper attempt'} · ${nameRole(c0, roleOf(c0))}`, `Round ${rnum(R)} · ${src}`,
        `<h4>1 · Packet as sent (attempt 1)</h4><div class="pk"><div class="f"><span class="lbl n">nonce</span><span class="hex ct">${esc(T.nonce || '–')}</span></div><div class="f"><span class="lbl c">ciphertext</span><span class="hex ct">${esc(words(before, 12))} …</span></div><div class="note">first 48 B of ${fmtInt(T.bytes)} B · hover to read</div></div>
         ${rv(4.4, `<h4>2 · As received after the interception</h4><div class="hex">${hexMark(after, diff)}</div><div class="note">${diff >= 0 ? `ciphertext byte ${fb} changed: ${before.substr(diff, 2)} → ${after.substr(diff, 2)} (${bits()} flipped)` : `ciphertext byte ${fb ?? '?'} changed (beyond the first 48 B shown)`}</div>`)}
         <div class="step rv fail" data-at="7.6"><span class="d">✗</span><div><b>GCM tag check</b> <span class="bad">${esc(err)}</span><div class="note" style="margin:0">dropped: not decrypted, not verified, not aggregated</div></div></div>
         ${hasRe ? `
         ${rv(X.nack[0], `<h4>3 · NACK · server → ${esc(droneName(c0))} · ${fmtInt(NK.bytes)} B</h4><div class="kv"><span class="k">plaintext</span><span class="hex">${esc(nackBody)}</span><span class="k">sealed with</span><span>control key of this link + server’s Ed25519 signature</span></div><div class="note" style="margin-top:2px">"round" counts from 0 (${T.round} = round ${T.round + 1})</div><div class="pk" style="margin-top:4px"><div class="f"><span class="lbl n">nonce</span><span class="hex ct">${esc(NK.nonce)}</span></div><div class="f"><span class="lbl c">ciphertext</span><span class="hex ct">${esc(words(NK.hex, 12))}</span></div></div>`)}
         ${rv(X.seal[0], `<h4>4 · Re-sealed: same update bytes, fresh nonce</h4><table class="t"><tr><th></th><th>nonce</th><th>ciphertext (first 8 B)</th></tr><tr><td class="dim">try 1</td><td class="m dim"><s>${esc(T.nonce)}</s></td><td class="m dim">${esc(before.slice(0, 16))}</td></tr><tr class="hl"><td>try 2</td><td class="m">${esc(RS.nonce)}</td><td class="m">${esc(RS.ch.slice(0, 16))}</td></tr></table><div class="note">A nonce is never reused with the same key; the new ciphertext shares nothing visible with the old one.</div>`)}
         <div class="step rv done" data-at="${X.ok}"><span class="d">✓</span><div><b>Tag ✓ · decrypt ✓ · Ed25519 ✓</b> on try ${RS.attempts || 2}</div></div>
         ${rv(X.ok + 0.6, cmp ? `<table class="t cmp"><tr><th>#</th><th>sent</th><th>decrypted</th><th></th></tr>${cmp}</table><div style="margin-top:6px">${RS.same ? '<span class="badge ok">bit-identical ✓</span>' : '<span class="badge bad">mismatch</span>'}</div>` : '')}
         ${rv(X.cont, `<h4>5 · Round ${rnum(R)} continues</h4><div style="margin:4px 0"><span class="badge ok">update not lost</span> <span class="badge acc">${N} of ${N} updates in</span></div><div class="kv"><span class="k">${esc(droneName(c0))} at the filter</span><span>${kept ? 'kept' : 'held back'}${st0 ? ` · ${esc(st0)}` : ''}</span><span class="k">Re-sends this round</span><span class="m">${R.rt || 1}</span></div><div class="note">A packet rejected more than security.max_retries times (default 3) stops the job with an error: an update is never silently dropped.</div>`)}`
         : rv(10, `<div style="margin:8px 0"><span class="badge bad">packet rejected</span></div><div class="note">Without the session key an attacker cannot make a valid tag, so it can neither read nor alter an update without being caught.</div>`)}`);
    },
    frame(t){
      const done1 = t >= 7.6;
      cids.forEach((c, i) => {
        if (c === c0){
          const f = t < 3.3 ? seg(t, 0.8, 3.3)*0.5 : t < 5.6 ? 0.5 : 0.5 + seg(t, 5.6, 7.6)*0.5;
          if (f <= 0) show(pk[c], L.A, L.ang); else if (f >= 1){ if (hasRe && t < X.nack[0]) show(pk[c], L.B, L.ang); else hide(pk[c]); } else { const p = lerp(L.A, L.B, f); show(pk[c], p, L.ang); }
          setA(cache, pk[c], 'class', 'pkt ' + (t >= 4.4 ? 'bad' : 'up'));
          if (!hasRe){
            slot(c, done1 ? 'bad' : ''); link(c, done1 ? 'rej' : 'sec');
            drone(c, {op: 1, st: done1 ? 'update rejected' : t >= 0.8 ? 'sending…' : 'ready', cls: done1 ? 'bad' : ''});
            return;
          }
          down(nk, c, seg(t, X.nack[0], X.nack[1]));
          up(rs, c, seg(t, X.fly[0], X.fly[1]));
          const ok = t >= X.ok, sealing = t >= X.seal[0] && t < X.seal[1];
          slot(c, ok ? 'ok' : t >= X.nack[0] ? 'nack' : done1 ? 'bad' : '');
          link(c, t >= X.cont ? (kept ? 'sel' : 'held') : ok ? 'sec' : t >= X.nack[0] ? 'resend' : done1 ? 'rej' : 'sec', t >= X.cont && kept ? 1.8 : t >= X.nack[0] && !ok ? 1.8 : 1.2);
          const st = t >= X.down[1] ? `model r${rnum(R)} ✓` : ok ? 'verified ✓ (try 2)' : t >= X.fly[0] ? 're-sending… try 2' : sealing ? 're-encrypting (new nonce)' : t >= X.nack[0] ? (t >= X.nack[1] ? 'NACK received' : 'NACK on its way') : done1 ? 'rejected · ' + err : t >= 0.8 ? 'sending…' : 'ready';
          drone(c, {op: 1, ring: sealing ? seg(t, X.seal[0], X.seal[1]) : null, st, cls: ok ? 'ok' : t >= X.nack[0] ? 'nack' : done1 ? 'bad' : ''});
        } else {
          const a = 0.3 + i*0.18, f = seg(t, a, a + 2.2), k = R.sel.includes(c);
          if (f <= 0) show(pk[c], links[c].A, links[c].ang); else up(pk[c], c, f);
          slot(c, hasRe && t >= X.cont ? (k ? 'sel' : 'held') : t >= a + 2.2 ? 'ok' : '');
          if (hasRe) link(c, t >= X.cont ? (k ? 'sel' : 'held') : 'sec', t >= X.cont && k ? 1.8 : 1.2);
          drone(c, {op: 1, st: hasRe && t >= X.down[1] ? `model r${rnum(R)} ✓` : hasRe && t >= X.cont ? (k ? 'kept' : 'held back') : t >= a + 2.2 ? 'verified ✓' : 'sending…',
            cls: hasRe && t >= X.cont ? (k ? 'acc' : 'warn') : t >= a + 2.2 ? 'ok' : ''});
        }
        if (hasRe) down(dl[c], c, seg(t, X.down[0] + (i % 5)*0.08, X.down[1]));
      });
      att.setAttribute('opacity', hasRe && t >= X.fly[0] ? 1 - seg(t, X.fly[0], X.fly[0] + 1) : seg(t, 1.6, 2.6));
      this.beam.setAttribute('opacity', t >= 3.3 && t < 5.6 ? (Math.floor(t*4) % 2 ? 1 : .4) : 0);
      this.al.setAttribute('opacity', hasRe && t >= X.nack[0] + 0.6 ? 0 : seg(t, 7.6, 8.0));
      srv.classList.toggle('alarm', t >= 7.6 && t < (hasRe ? X.nack[0] + 0.6 : 12));
      srv.classList.toggle('flash', hasRe && ((t >= X.ok && t < X.ok + 1) || (t >= X.cont && t < X.down[0])));
      srvSub.textContent = hasRe && t >= X.cont ? `aggregating round ${rnum(R)}` : '';
    }};
}
function sceneLiveNow(Ls, R){
  const st = byCid(Ls.drones);
  return {step: null, key: 'now', title: 'Now', dur: 1,
    cues: [[0, `Live: round ${Ls.round != null ? Ls.round + 1 : '?'}${Ls.phase ? ' · ' + Ls.phase : ''}. Full colour = connected and finished local training; faded with a ring = still training; amber = re-sending after a rejected packet.`]],
    setup(){
      base(); hud({round: Ls.round != null ? Ls.round : (R ? R.round : 0)});
      this.nk = {};
      let rows = '';
      cids.forEach(c => { const x = st[c] || {}, rs = x.state === 'resending';
        if (rs) this.nk[c] = packet('nack', 're-send requested');
        rows += `<tr><td>${droneName(c)}</td><td class="${rs ? 'nackc' : ''}">${esc(x.state || '–')}${rs && x.dir ? ` · ${esc(x.dir)} try ${x.attempt || 2}` : ''}</td><td class="r">${x.progress != null ? Math.round(100*x.progress) + ' %' : ''}</td><td class="r">${x.ms != null ? (x.ms/1000).toFixed(2) + ' s' : ''}</td></tr>`; });
      panel('Live link state', `Round ${Ls.round != null ? Ls.round + 1 : '?'} · ${esc(Ls.phase || '')}`,
        `<table class="t"><tr><th>Drone</th><th>state</th><th class="r">progress</th><th class="r">local time</th></tr>${rows}</table>
         ${Ls.rt ? `<div class="note"><span class="badge nack">⟲ ${Ls.rt} re-send${Ls.rt > 1 ? 's' : ''} this round</span>${Ls.event ? ` last: ${esc(droneName(Ls.event.cid))} ${esc(Ls.event.dir)} (${esc(Ls.event.error)})` : ''}</div>` : ''}
         <div class="note">Refreshes every 2 s while training. Pick a step above to replay the last finished round.</div>`);
    },
    frame(t){  // static picture of the logged state (no animation loop while idle)
      cids.forEach(c => {
        const x = st[c] || {}, s = (x.state || '').toLowerCase();
        const training = s.startsWith('train'), sent = /sent|received|uploaded|verified|done/.test(s), rs = s === 'resending';
        drone(c, {op: training ? .42 : x.state ? 1 : .6, ring: training ? (x.progress ?? 0.05) : null, spin: 0,
          st: training ? `training… ${x.progress != null ? Math.round(100*x.progress) + '%' : ''}` : rs ? `re-sending · ${x.dir || ''} try ${x.attempt || 2}` : (x.state || ''),
          cls: rs ? 'nack' : sent || s === 'ready' ? 'ok' : ''});
        slot(c, rs ? 'nack' : sent ? 'ok' : '');
        link(c, rs ? 'resend' : 'sec', rs ? 1.8 : 1.2);
        if (rs && this.nk[c]){ const back = x.dir === 'uplink' || x.dir === 'ballot'; (back ? down : up)(this.nk[c], c, 0.5); }
      });
    }};
}

// ---------- timeline ----------
if (D.mode === 'demo'){
  scenes.push(sceneEnroll(), sceneTrain(R0), sceneEncrypt(R0), sceneTransmit(R0), sceneVerify(R0), sceneFilter(R0));
  if (RV){
    if (RV.round - R0.round > 1) scenes.push(sceneFF(R0, RV, 'Fast-forward'));
    scenes.push(sceneVote(RV), sceneDQN(RV), sceneAggregate(RV));
  } else scenes.push(sceneAggregate(R0));
  if (D.tamper && RTm){
    const RA = RV || R0;
    if (RTm.round - RA.round > 1) scenes.push(sceneFF(RA, RTm, 'Fast-forward'));
    scenes.push(sceneTamper(RTm, D.tamper));
  }
  // dashboard control that runs each step in a real (non-demo) training run
  const vsTxt = vs != null ? ` (this run: from round ${vs + 1} of ${RT})` : '';
  Object.assign(REAL, {
    enroll: 'sidebar <b>New experiment</b> → <b>Start training</b>: every dashboard run uses real mTLS (gRPC over TLS 1.3, X.509 certificates) and each drone process enrolls fresh signing keys before round 1.',
    train: '<b>New experiment</b> → “Drones”, “FL rounds”, “Local SGD steps / round” → <b>Start training</b>; follow it live in <b>Secure Link → This run’s link</b>.',
    encrypt: 'every update from the <b>New experiment</b> training form is signed and encrypted like this by the drone process before it leaves the drone.',
    transmit: 'every <b>New experiment</b> run (real mTLS); packet sizes and simulated 10 Mbit/s transfer times are logged per round (securelink.jsonl).',
    verify: 'every <b>New experiment</b> run (real mTLS); the run checks every decrypted model is bit-identical and stops if not.',
    filter: 'Methods → <b>Jiang, fixed weight</b> or <b>Jiang + DQN</b>; “Filter keeps K models” sets K.',
    vote: 'Jiang methods; <b>“Voting starts at (fraction of rounds)”</b>' + vsTxt + '. Vote copies and ballots use the encrypted link.',
    dqn: 'Methods → <b>Jiang + DQN</b> (or a pre-trained DQN); <b>Jiang, fixed weight</b> uses 1/(K+1) instead.',
    aggregate: 'every round of every method; the broadcast is encrypted per drone in every mTLS training run.',
    tamper: 'also set <b>“Simulate interceptions (probability)”</b> in the mTLS training form (security.tamper_prob): a simulated attacker alters each packet with that probability; a rejected packet is NACKed and the drone re-sends it, up to security.max_retries (default 3) times.'});
} else {
  const R = RR.cur;
  if (D.live && D.live.drones && D.live.drones.length) scenes.push(sceneLiveNow(D.live, R));
  if (R){
    if (R.local.length) scenes.push(sceneTrain(R));
    if (R.up.length) scenes.push(sceneEncrypt(R), sceneTransmit(R), sceneVerify(R));
    if (R.rows.length && R.rows.some(x => x.score != null)) scenes.push(sceneFilter(R));
    if (R.voted.length) scenes.push(sceneVote(R));
    if (R.rows.length && vs != null && R.round >= vs) scenes.push(sceneDQN(R));
    if (R.down.length || R.rows.length) scenes.push(sceneAggregate(R));
    if (D.tamper && D.tamper.round === R.round) scenes.push(sceneTamper(R, D.tamper));
  }
  if (!scenes.length){ insp.innerHTML = '<div class="note">Nothing logged yet.</div>'; }
}
const starts = []; let total = 0; scenes.forEach(s => { starts.push(total); total += s.dur; });

// chips: one per demo step (1-10) or one per scene in the live view
const chips = $('#chips');
const SHORT = {now: 'Now (live)', train: 'Local training', encrypt: 'Encryption', transmit: 'Transmission', verify: 'Decrypt & verify',
  filter: 'Filter', vote: 'Vote', dqn: 'DQN weight', aggregate: 'Aggregate & broadcast', tamper: 'Interception → re-send'};
const chipDefs = D.mode === 'demo'
  ? [[1, 'Enrollment & keys'], [2, 'Local training'], [3, 'Encryption'], [4, 'Transmission'], [5, 'Decrypt & verify'], [6, 'Filter'], [7, 'Vote'], [8, 'DQN weight'], [9, 'Aggregate & broadcast'], [10, 'Interception → re-send']]
      .map(([n, t]) => ({n, t, i: scenes.findIndex(s => s.step === n)})).filter(x => x.i >= 0)
  : scenes.map((s, i) => ({n: i + 1, t: SHORT[s.key] || s.title, i}));
// hover card per step: which real round it shows, why that round, and context not visible in the scene
const tipEl = document.createElement('div'); tipEl.id = 'chiptip'; document.body.appendChild(tipEl);
function stepInfo(n){
  if (D.mode !== 'demo') return '';
  const I = D.info || {}, T = RT, vsr = vs != null ? vs + 1 : null, src = I.run ? `<div class="m">Source: ${esc(I.run)}</div>` : '';
  const voteTxt = vsr ? `Voting starts at round <b>${vsr}</b> (${Math.round((I.vfrac || 0) * 100)}% of ${T} rounds); before that the filter alone decides.` : '';
  const mb = R => R && R.totals && R.totals.bytes ? (R.totals.bytes / 1e6).toFixed(1) + ' MB' : '–';
  const ms = (R0 && R0.local || []).map(x => x.ms).filter(x => x != null);
  const r1 = `<div class="r">Round ${rnum(R0)} of ${T}</div>`;
  switch (n){
    case 1: return `<div class="r">Once, before round 1</div>The operator's CA enrolls the ${I.nc || 10} drones and the server: each gets an Ed25519 signing key and an X25519 key; fingerprints shown are the real ones of this run.${src}`;
    case 2: return `${r1}The first round of the run. Every drone trains ${I.steps || '–'} local steps from the same global model; real times ${ms.length ? (Math.max(...ms) < 1000 ? Math.round(Math.min(...ms)) + '–' + Math.round(Math.max(...ms)) + ' ms' : (Math.min(...ms)/1000).toFixed(2) + '–' + (Math.max(...ms)/1000).toFixed(2) + ' s') : '–'} per drone (the animation slows this down so it can be seen). ${voteTxt}${src}`;
    case 3: return `${r1}Each update (${R0 && R0.up[0] ? R0.up[0].bytes.toLocaleString() : '–'} bytes) is signed, then encrypted. The panel shows the first 8 weights and first 48 bytes; click another drone to see its packet.${src}`;
    case 4: return `${r1}${R0 ? R0.up.length : '–'} uploads, ${mb(R0)} this round in total. Transfer time is simulated for a ${D.crypto.bandwidth_mbps || 10} Mbit/s, ${D.crypto.latency_ms || 20} ms link (computed, not waited for).${src}`;
    case 5: return `${r1}The server checks every packet's tag, decrypts it and verifies the signature against the drone's enrolled key; the decrypted values are compared bit for bit with what the drone sent.${src}`;
    case 6: return `${r1}The filter keeps the ${I.K || 5} updates closest to the others (summed squared distance) and drops the rest. ${voteTxt}${src}`;
    case 7: return RV ? `<div class="r">Round ${rnum(RV)} of ${T}</div>${voteTxt} ${RV.round === vs ? 'This is that first voting round.' : `Round ${rnum(RV)} is the first voting round in which a dropped update was voted back in, so steps 7–9 have something to show.`} Each honest drone tests the dropped updates on its own data; more than half yes = readmitted. The ${I.np ?? 3} poisoned drones vote at random.${src}` : '';
    case 8: return RV ? `<div class="r">Round ${rnum(RV)} of ${T} (same round as the vote)</div>The DQN picks a weight for each readmitted update (0 to 3× the fixed 1/(K+1)), from the round, freshness, public-exam score and vote share.${src}` : '';
    case 9: return RV ? `<div class="r">Round ${rnum(RV)} of ${T} (same round)</div>New global = filtered model plus the weighted readmitted updates; it goes back to all ${I.nc || 10} drones encrypted (${RV.down.length} downlinks). Round traffic ${mb(RV)}.${src}` : '';
    case 10: return RTm && D.tamper ? `<div class="r">Round ${rnum(RTm)} of ${T}</div>A real interception from this run: ${I.tprob ? `about ${(I.tprob * 100).toFixed(1)}% of packets were altered on purpose (simulated)` : 'simulated'}, ${I.rt_total || 0} re-sends in the whole run. Here ${droneName(D.tamper.cid)}'s packet was rejected and re-sent (up to ${I.maxret || 3} tries allowed).${src}` : '';
  }
  return '';
}
chipDefs.forEach(d => { const b = document.createElement('button'); b.className = 'chip'; b.innerHTML = `<span class="n">${d.n}</span>${esc(d.t)}`;
  b.onclick = () => { mode = 'single'; enter(d.i, true); }; d.el = b; chips.appendChild(b);
  const html = stepInfo(d.n);
  if (html){
    b.addEventListener('mouseenter', () => { tipEl.innerHTML = html; tipEl.style.display = 'block';
      const r = b.getBoundingClientRect(), w = tipEl.offsetWidth;
      tipEl.style.left = Math.max(6, Math.min(r.left, window.innerWidth - w - 6)) + 'px'; tipEl.style.top = (r.bottom + 6) + 'px'; });
    b.addEventListener('mouseleave', () => { tipEl.style.display = 'none'; });
  }
});
if (D.mode !== 'demo') $('#playall span').textContent = D.live ? 'Play last round' : 'Play this round';
const prog = $('#prog');
starts.forEach((s, i) => { if (i){ const k = document.createElement('div'); k.className = 'tick'; k.style.left = (100*s/total) + '%'; prog.appendChild(k); } });

// ---------- player ----------
// The animation loop runs only while playing AND the page is visible AND the frame is on screen; it draws at most
// ~30 fps (a timer between animation frames). Paused, finished, idle, hidden or scrolled away: no callbacks at all.
let cur = 0, t = 0, playing = false, speed = 1, mode = 'demo', lastNow = null, lastCue = null, lastHow = null;
let raf = null, timer = null, visible = !document.hidden, onscreen = true;
const FRAME_MS = 33;
const want = () => playing && visible && onscreen && scenes.length > 0;
function halt(){ if (raf != null) cancelAnimationFrame(raf); if (timer != null) clearTimeout(timer); raf = timer = null; lastNow = null; }
function kick(){ if (!want()){ halt(); return; } if (raf == null && timer == null){ lastNow = null; raf = requestAnimationFrame(loop); } }
function enter(i, play){
  cur = clamp(i, 0, scenes.length - 1); t = 0; lastCue = null;
  scenes[cur].setup(); scenes[cur].frame(0); reveal(0); playing = !!play; ui(); kick();
}
function cue(){
  const s = scenes[cur]; let txt = '';
  for (const [a, x] of s.cues) if (t >= a) txt = x;
  if (txt !== lastCue){ lastCue = txt; $('#capt').textContent = txt; }
  $('#capk').textContent = (D.mode === 'demo' && s.step ? `Step ${s.step} · ` : '') + s.title;
  const how = D.mode === 'demo' && REAL[s.key] ? '<b>In a real run:</b> ' + REAL[s.key] : '';
  if (how !== lastHow){ lastHow = how; $('#caph').innerHTML = how; }
}
const mmss = x => Math.floor(x/60) + ':' + pad2(Math.floor(x % 60));
function ui(){
  const s = scenes[cur]; if (!s) return;
  const span = mode === 'demo' ? total : s.dur, pos = mode === 'demo' ? starts[cur] + t : t;
  prog.querySelector('.fill').style.width = (100*pos/(span || 1)).toFixed(2) + '%';
  prog.querySelectorAll('.tick').forEach(e => e.style.display = mode === 'demo' ? '' : 'none');
  $('#clock').textContent = mmss(pos) + ' / ' + mmss(span);
  chipDefs.forEach(d => { d.el.classList.toggle('on', d.i === cur || (scenes[cur].step == null && false));
    d.el.classList.toggle('done', mode === 'demo' && d.i < cur); });
  $('#pauseic').innerHTML = playing ? '<rect x="2.5" y="1.5" width="2.6" height="9" rx=".6"/><rect x="6.9" y="1.5" width="2.6" height="9" rx=".6"/>' : '<path d="M2.5 1.5v9l8-4.5z"/>';
  cue();
}
function loop(now){
  raf = null;
  if (!want()){ lastNow = null; return; }
  if (lastNow == null) lastNow = now;
  const dt = Math.min(0.1, (now - lastNow)/1000); lastNow = now;
  t += dt*speed;
  const s = scenes[cur];
  if (t >= s.dur){
    if (mode === 'demo' && cur < scenes.length - 1){ const extra = t - s.dur; enter(cur + 1, true); t = extra; }
    else { t = s.dur; playing = false; }
  }
  scenes[cur].frame(t); reveal(t); ui();
  if (want() && raf == null && timer == null){
    const spent = performance.now() - now;
    timer = setTimeout(() => { timer = null; if (want()) raf = requestAnimationFrame(loop); else lastNow = null; }, Math.max(0, FRAME_MS - spent));
  } else if (!want()) halt();
}
document.addEventListener('visibilitychange', () => { visible = !document.hidden; kick(); });
if ('IntersectionObserver' in window){  // the frame scrolled away or sits in a hidden tab: stop drawing
  new IntersectionObserver(es => { onscreen = es.some(e => e.isIntersecting); kick(); }).observe(document.body);
}
$('#playall').onclick = () => { mode = 'demo'; enter(0, true); };
$('#pause').onclick = () => { if (!scenes.length) return; if (!playing && t >= scenes[cur].dur) enter(cur, true); else { playing = !playing; ui(); kick(); } };
$('#next').onclick = () => enter(cur + 1, true);
$('#prev').onclick = () => enter(t > 1.5 ? cur : cur - 1, true);
document.querySelectorAll('#speed button').forEach(b => b.onclick = () => { speed = +b.dataset.s; document.querySelectorAll('#speed button').forEach(x => x.classList.toggle('on', x === b)); });
prog.onclick = e => {
  const r = prog.getBoundingClientRect(), f = clamp((e.clientX - r.left)/r.width);
  if (mode === 'demo'){ const x = f*total; let i = starts.length - 1; while (i > 0 && starts[i] > x) i--; const was = playing; enter(i, was); t = x - starts[i]; }
  else t = f*scenes[cur].dur;
  scenes[cur].frame(t); reveal(t); ui();
};

if (scenes.length){
  if (D.mode === 'demo'){
    // opening frame: the network as it looks once enrolled, waiting for "Play"
    enter(0, false); scenes[0].frame(scenes[0].dur); reveal(scenes[0].dur); t = 0; ui();
    $('#capk').textContent = 'Secure link demo';
    $('#capt').textContent = `Real values from a ${RT}-round run. Press “Play full demo” (about ${Math.round(total/60*2)/2} min) or pick a step.`;
    lastHow = null; $('#caph').innerHTML = '<b>Only real workflows:</b> each step is code a dashboard training run executes; the step names the setting that turns it on.';
  } else {  // a static picture of the round until the viewer presses play
    mode = 'single'; enter(0, false);
    if (scenes[0].key !== 'now'){ scenes[0].frame(scenes[0].dur); reveal(scenes[0].dur); t = scenes[0].dur; ui(); }
    if (!D.live) mode = 'demo';
  }
}
})();
"""
