"""Aggregation Server panel of the dashboard (app.py): what the server can see.

  Top bar       dataset, run, state, FL round, model version, open-alert bell
  Trust scores  per-drone trust computed ONLY from server-visible signals in rounds.jsonl (filter decision, vote,
                filter distance); formula shown on the page
  Audit log     each round's server decisions in plain words, filterable by drone / decision / round

The server never sees raw traffic or true roles, so nothing on these pages uses the `role` field, except to colour
drone names (poisoned red, rare green) when the sidebar switch "Colour drones by true role (simulation view)" is on.
Read-only: nothing is written to runs/, data/ or data_cache/.
"""
import json
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import app_ids
import app_roles

# Set once by app.py (configure): its decision names/colours and role names (role names are used on the Drone Sites
# pages only, where the site owner knows its own role).
STATUS, ROLE_NAMES, ROLE_COLORS = {}, {}, {}
_HELP = [""]


def configure(status, status_help, role_names, role_colors):
    STATUS.update(status)
    _HELP[0] = status_help
    # role dicts are kept by reference: app.set_dataset renames the rare role per run
    globals()["ROLE_NAMES"], globals()["ROLE_COLORS"] = role_names, role_colors


def drone_name(cid):
    return f"Drone {cid + 1:02d}"


def true_roles(cf):
    """{cid: role} for colouring drone names on the server pages: only in the simulation view (sidebar switch on);
    off = {} (names uncoloured, as the real server knows no roles). Never used in any computation."""
    if not app_roles.server_on() or cf.empty or "role" not in cf:
        return {}
    return cf.groupby("cid")["role"].last().to_dict()


def status_label(status, rnd, vote_start):
    """Display name of a decision; 'flagged' before voting starts = dropped while voting is frozen (as in app.py)."""
    key = "frozen" if status == "flagged" and rnd < vote_start else status
    return STATUS.get(key, (key, "#999"))[0]


# ---------- server-visible rows ----------
COLS = ["round", "cid", "role", "status", "weight", "score", "filtered_out", "votes_yes", "votes_total", "readmitted",
        "dqn_weight"]


@st.cache_data(max_entries=64, show_spinner=False)
def _client_frame(path, mtime, size):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # half-written last line while training
                    break
                for c in r.get("clients", []):
                    rows.append([r["round"], c["cid"], c.get("role"), c.get("status"), c.get("weight", 0.0),
                                 c.get("score"), c.get("filtered_out"), c.get("votes_yes"), c.get("votes_total"),
                                 c.get("readmitted"), c.get("dqn_weight")])
    except OSError:
        pass
    df = pd.DataFrame(rows, columns=COLS)
    for k in ("score", "votes_yes", "votes_total", "dqn_weight", "weight"):
        df[k] = pd.to_numeric(df[k], errors="coerce")
    return df


def client_frame(run_dir, method, seed):
    """One row per drone per round from rounds.jsonl (cached until the file changes)."""
    path = os.path.join(run_dir, method, f"seed_{seed}", "rounds.jsonl")
    try:
        stt = os.stat(path)
    except OSError:
        return pd.DataFrame(columns=COLS)
    return _client_frame(path, stt.st_mtime, stt.st_size)


# ---------- trust score ----------
TRUST_ALPHA = 0.1   # EWMA smoothing (notes/SPEC_P17.md)
TRUST_MIX = 0.7     # share of the decision signal t; the rest is the distance term (1 - percentile)
TRUST_START = 0.5   # every drone starts at 50 (Watch): trust is earned
BANDS = [(70, "Trusted", "#3f9f4f"), (40, "Watch", "#c99a06"), (0, "Untrusted", "#d64545")]
TRUST_HELP = ("Server-side trust (0-100) from what the server observes only: filter decision, vote outcome and filter "
              "distance; EWMA with α = 0.1. ≥ 70 Trusted · 40-70 Watch · < 40 Untrusted.")
VOTING = ("jiang",)  # methods with a voting layer: a dropped model is 'pending' until the vote


def band(v):
    return next(name for lo, name, _ in BANDS if v >= lo)


def band_color(name):
    return next(c for _, n, c in BANDS if n == name)


def decision_signal(status, voting):
    """t per drone per round: 1 = accepted by the filter or voted back in, 0 = voted out (or dropped for good by a
    method without a vote), 0.5 = dropped and the vote is still pending (incl. before voting starts)."""
    if status in ("accepted", "readmitted", "vetoed", "averaged"):
        return 1.0
    if status == "flagged":
        return 0.5 if voting else 0.0
    return 0.0  # rejected, excluded


def trust_from_frame(cf, method):
    if cf.empty:
        return pd.DataFrame(columns=["round", "cid", "t", "pct", "s", "trust"])
    voting = method.startswith(VOTING)
    df = cf[["round", "cid", "status", "score"]].copy()
    df["t"] = [decision_signal(s, voting) for s in df["status"]]
    # distance percentile among this round's models: 0 = most central, 1 = farthest (ties share the average rank)
    n = df.groupby("round")["cid"].transform("count")
    rk = df.groupby("round")["score"].rank(method="average")
    df["pct"] = ((rk - 1) / (n - 1).clip(lower=1)).where(df["score"].notna())
    has_d = df["pct"].notna()
    df["s"] = np.where(has_d, TRUST_MIX * df["t"] + (1 - TRUST_MIX) * (1 - df["pct"].fillna(0)), df["t"])
    df = df.sort_values(["cid", "round"])
    out = []
    for cid, g in df.groupby("cid", sort=True):
        e, vals = TRUST_START, []
        for s in g["s"].to_numpy():
            e = (1 - TRUST_ALPHA) * e + TRUST_ALPHA * s
            vals.append(100 * e)
        out.append(g.assign(trust=vals))
    return pd.concat(out).sort_values(["round", "cid"]).reset_index(drop=True)


@st.cache_data(max_entries=64, show_spinner=False)
def _trust_cached(path, mtime, size, method):
    return trust_from_frame(_client_frame(path, mtime, size), method)


def trust_frame(run_dir, method, seed):
    path = os.path.join(run_dir, method, f"seed_{seed}", "rounds.jsonl")
    try:
        stt = os.stat(path)
    except OSError:
        return trust_from_frame(pd.DataFrame(columns=COLS), method)
    return _trust_cached(path, stt.st_mtime, stt.st_size, method)


def trust_chart(tr, meta, single=False, roles=None):
    """Trust over rounds with the three bands shaded and the vote start marked."""
    last = int(tr["round"].max()) + 1 if not tr.empty else 1
    bands = pd.DataFrame([{"lo": 0, "hi": 40, "band": "Untrusted"}, {"lo": 40, "hi": 70, "band": "Watch"},
                          {"lo": 70, "hi": 100, "band": "Trusted"}])
    shade = alt.Chart(bands).mark_rect(opacity=0.09).encode(
        y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 100])), y2="hi:Q",
        color=alt.Color("band:N", scale=alt.Scale(domain=[b[1] for b in BANDS], range=[b[2] for b in BANDS]),
                        legend=None))
    d = tr.assign(Round=tr["round"] + 1)
    step = max(1, len(d["Round"].unique()) // 400)  # keep the chart light on 750-round runs
    d = d[(d["Round"] % step == 0) | (d["Round"] == last)]
    enc = dict(x=alt.X("Round:Q", title="FL round", scale=alt.Scale(domain=[1, last])),
               y=alt.Y("trust:Q", title="Server trust", scale=alt.Scale(domain=[0, 100])),
               tooltip=["drone:N", "Round:Q", alt.Tooltip("trust:Q", format=".1f")])
    if single:
        line = alt.Chart(d).mark_line(strokeWidth=2.2, color=app_ids.ink()).encode(**enc)
    else:
        sel = alt.selection_point(fields=["drone"], bind="legend")
        # line colours follow the role colours: liars in reds, the rare drone green, regular drones in blues/greys
        # (no red/green for regular drones, so a colour never suggests a role it doesn't have)
        rmap = {drone_name(c): r for c, r in (roles or {}).items()}
        reds, greens = ["#c62828", "#e57373", "#8e0000", "#ff8a80"], ["#2e7d32", "#81c784"]
        neutral = ["#1565c0", "#90a4ae", "#5e35b1", "#00838f", "#455a64", "#7986cb", "#4fc3f7", "#6d4c41", "#b0bec5", "#283593"]
        drones = sorted(d["drone"].unique())
        it = {"poisoned": iter(reds * 3), "rare": iter(greens * 5), None: iter(neutral * 3)}
        rng = [next(it[app_roles.norm(rmap.get(x)) if app_roles.norm(rmap.get(x)) in ("poisoned", "rare") else None])
               for x in drones]
        line = alt.Chart(d).mark_line(strokeWidth=1.8).encode(
            color=alt.Color("drone:N", scale=alt.Scale(domain=drones, range=rng),
                            legend=alt.Legend(orient="right", title="Click to highlight", symbolType="stroke",
                                                         labelColor=app_roles.label_color(
                                                             {drone_name(c): r for c, r in (roles or {}).items()}))),
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.12)), **enc).add_params(sel)
    vs = meta.get("vote_start_round", 0) + 1
    rule = alt.Chart(pd.DataFrame({"x": [vs]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
    # band shading and drone lines each keep their own colour scale (a shared one hid the lines)
    return (shade + line + rule).resolve_scale(color="independent").properties(height=200 if single else 300)


def tab_trust(run_dir, meta, methods, seeds):
    app_ids.note("How much the aggregation server trusts each drone, from what the server itself observes: whether "
                 "the filter kept its update, how the swarm voted on it, and how far it lay from the others. The true "
                 "roles are never used.")
    c1, c2 = st.columns([3, 1])
    srv = [m for m in methods if m != "fedavg_clean"] or methods
    with c1:
        method = st.radio("Method", srv, horizontal=True, format_func=app_ids.lab, key="tr_m",
                          index=next((i for i, m in enumerate(srv) if m.startswith("jiang_dqn")), 0))
    with c2:
        seed = st.radio("Seed", seeds, horizontal=True, key="tr_s")
    with st.container(border=True):
        st.markdown("**Trust formula** (per drone, updated every round; nothing hidden)")
        st.latex(r"\text{trust}_r = 100 \cdot \mathrm{EWMA}_{\alpha=0.1}\big(" + f"{TRUST_MIX}" + r"\,t_r + "
                 + f"{1 - TRUST_MIX:.1f}" + r"\,(1 - p_r)\big),\quad \text{start} = 50")
        st.markdown(
            "<div class='ids-kv' style='line-height:1.7'>"
            "<b>t<sub>r</sub></b> = 1 selected by the filter or voted back in · 0 voted out · 0.5 dropped, vote pending "
            "(also while voting has not started; for methods without a vote a dropped update counts 0, FedAvg-style "
            "averaging counts 1)<br>"
            "<b>p<sub>r</sub></b> = filter-distance percentile among this round's updates (0 = most central, 1 = farthest; "
            "left out when the method has no filter, then trust = EWMA of t)<br>"
            "EWMA: trust<sub>r</sub> = 0.9 · trust<sub>r−1</sub> + 0.1 · signal<sub>r</sub> · bands: "
            f"<span style='color:{BANDS[0][2]}'>≥ 70 Trusted</span> · <span style='color:{BANDS[1][2]}'>40-70 Watch</span>"
            f" · <span style='color:{BANDS[2][2]}'>&lt; 40 Untrusted</span></div>", unsafe_allow_html=True)
    tr = trust_frame(run_dir, method, seed)
    if tr.empty:
        st.info("No rounds yet for this method and seed.")
        return
    cf = client_frame(run_dir, method, seed)
    roles = true_roles(cf)
    last = int(tr["round"].max()) + 1
    follow = st.toggle("Latest round", True, key="tr_follow")
    upto = last if follow or last == 1 else st.slider("As of round", 1, last, last, key="tr_round")
    tr_u = tr[tr["round"] < upto]
    cf_u = cf[cf["round"] < upto]
    now = tr_u[tr_u["round"] == upto - 1].set_index("cid")
    vs = meta.get("vote_start_round", 0)
    k = st.columns(4)
    bands_now = [band(v) for v in now["trust"]]
    for col, (_, name, color) in zip(k, BANDS):
        col.metric(name, f"{bands_now.count(name)} drones")
    k[3].metric("Round", f"{upto} / {meta.get('rounds', last)}")

    rows = []
    for cid, row in now.iterrows():
        g = cf_u[cf_u["cid"] == cid]
        h = tr_u[tr_u["cid"] == cid]
        step = max(1, len(h) // 80)
        last_status = g.iloc[-1] if len(g) else None
        voted = g[g["votes_total"].notna()]
        rows.append({"Drone": drone_name(cid), "Trust": float(row["trust"]), "Band": band(row["trust"]),
                     "Trend": [round(float(x), 1) for x in h["trust"].iloc[::step]],
                     "Selected by filter": float((g["status"] == "accepted").mean()) if len(g) else np.nan,
                     "Voted back in": int((voted["readmitted"] == True).sum()),  # noqa: E712 (column holds bools/None)
                     "Voted out": int((g["status"] == "rejected").sum()),
                     "Median distance pct": float(h["pct"].median()) if h["pct"].notna().any() else np.nan,
                     "Last decision": status_label(last_status["status"], int(last_status["round"]), vs)
                     if last_status is not None else "–"})
    tbl = pd.DataFrame(rows)
    sty = tbl.style.map(lambda b: f"color:{band_color(b)};font-weight:700" if b else "", subset=["Band"])
    st.dataframe(app_roles.rows(sty, [roles.get(c) for c in now.index], ["Drone"])
                 .format({"Selected by filter": "{:.0%}", "Median distance pct": "{:.2f}"}, na_rep="–"),
                 hide_index=True, width="stretch",
                 column_config={"Trust": st.column_config.ProgressColumn("Trust", min_value=0, max_value=100, format="%.0f"),
                                "Trend": st.column_config.LineChartColumn("Trust over rounds", y_min=0, y_max=100),
                                "Drone": st.column_config.TextColumn("Drone")})
    st.altair_chart(trust_chart(tr_u.assign(drone=[drone_name(c) for c in tr_u["cid"]]), meta, roles=roles),
                    width="stretch")
    app_ids.note("Dashed line = voting starts. Before it, a dropped update is pending (0.5) so trust stays near the "
                 "middle; once votes are cast, updates voted out pull trust down and accepted ones pull it up. Click a "
                 "drone in the legend to highlight it. Ground truth (which drones are really poisoned) is only on the "
                 "Drone Sites pages.")


# ---------- audit log ----------
def _phrase(r, vs, voting):
    st_, sc = r["status"], r["score"]
    dist = f" (distance {sc:.2f})" if pd.notna(sc) else ""
    votes = f"{int(r['votes_yes'])}/{int(r['votes_total'])} yes" if pd.notna(r["votes_total"]) else "vote"
    w = f"weight {r['weight']:.1%}"
    if st_ == "accepted":
        return f"selected by filter{dist} → {w}"
    if st_ == "flagged":
        if not voting:
            return f"dropped by filter{dist} → excluded"
        return f"dropped by filter{dist} → " + ("voting not started" if r["round"] < vs else "vote pending")
    if st_ == "readmitted":
        dq = f" → DQN weight {r['dqn_weight']:g}" if pd.notna(r["dqn_weight"]) else ""
        return f"dropped by filter{dist} → voted back in ({votes}){dq} → {w}"
    if st_ == "vetoed":
        return f"dropped by filter{dist} → voted back in ({votes}) → DQN weight 0"
    if st_ == "rejected":
        return f"dropped by filter{dist} → voted out ({votes})"
    if st_ == "averaged":
        return f"averaged (no filter) → {w}"
    if st_ == "excluded":
        return "left out (reference run without attackers)"
    return st_


def tab_audit(run_dir, meta, methods, seeds):
    app_ids.note("Every decision the aggregation server took, round by round, in plain words: what the filter did "
                 "with each drone's update, how the swarm voted, and the weight it got in the global model.")
    c1, c2 = st.columns([3, 1])
    srv = [m for m in methods if m != "fedavg_clean"] or methods
    with c1:
        method = st.radio("Method", srv, horizontal=True, format_func=app_ids.lab, key="au_m",
                          index=next((i for i, m in enumerate(srv) if m.startswith("jiang_dqn")), 0))
    with c2:
        seed = st.radio("Seed", seeds, horizontal=True, key="au_s")
    cf = client_frame(run_dir, method, seed)
    if cf.empty:
        st.info("No rounds yet for this method and seed.")
        return
    vs = meta.get("vote_start_round", 0)
    voting = method.startswith(VOTING)
    df = cf.assign(Decision=[status_label(s, r, vs) for s, r in zip(cf["status"], cf["round"])])
    last = int(df["round"].max()) + 1
    f = st.columns([2, 3, 2])
    drones = sorted(df["cid"].unique())
    pick_d = f[0].multiselect("Drones", drones, [], format_func=drone_name, key="au_d", placeholder="All drones")
    kinds = [lbl for lbl, _ in STATUS.values() if lbl in set(df["Decision"])]
    pick_k = f[1].multiselect("Decision", kinds, [], key="au_k", placeholder="All decisions")
    lo, hi = f[2].slider("Rounds", 1, last, (max(1, last - 49), last), key="au_r") if last > 1 else (1, 1)
    v = df[(df["round"] + 1 >= lo) & (df["round"] + 1 <= hi)]
    if pick_d:
        v = v[v["cid"].isin(pick_d)]
    if pick_k:
        v = v[v["Decision"].isin(pick_k)]
    v = v.sort_values(["round", "cid"], ascending=[False, True])
    msg = [f"Round {int(r['round']) + 1} · {drone_name(int(r['cid']))} {_phrase(r, vs, voting)}" for _, r in v.iterrows()]
    out = pd.DataFrame({"Event": msg, "Round": v["round"] + 1, "Drone": [drone_name(c) for c in v["cid"]],
                        "Decision": v["Decision"], "Weight": v["weight"], "Filter distance": v["score"],
                        "Votes yes": [f"{int(a)}/{int(b)}" if pd.notna(b) else "–"
                                      for a, b in zip(v["votes_yes"], v["votes_total"])]})
    counts = v["Decision"].value_counts()
    color_of = dict(STATUS.values())
    st.markdown(" &nbsp; ".join(f"<span class='ids-kv'><span class='ids-dot' style='background:{color_of.get(k_, '#999')}'>"
                                f"</span>{app_ids.esc(k_)} <b>{int(counts.get(k_, 0)):,}</b></span>" for k_ in kinds)
                + f"<span class='ids-kv'> &nbsp;· {len(v):,} events, rounds {lo}-{hi}, newest first</span>",
                unsafe_allow_html=True)
    roles = true_roles(cf)
    sty = out.style.map(lambda s: f"background-color:{color_of.get(s, '')};color:white;font-weight:600"
                        if s in color_of else "", subset=["Decision"])
    st.dataframe(app_roles.rows(sty, [roles.get(c) for c in v["cid"]], ["Event", "Drone"])
                 .format({"Weight": "{:.1%}", "Filter distance": "{:.3f}"}, na_rep="–"),
                 hide_index=True, width="stretch", height=min(560, 35 * (len(out) + 1) + 3),
                 column_config={"Event": st.column_config.TextColumn("Event", width="large")})
    st.download_button("Export CSV", out.to_csv(index=False), f"audit_{method}_seed{seed}.csv", "text/csv",
                       key="au_csv")
    st.caption(_HELP[0])


# ---------- top bar ----------
def open_alerts(run):
    """Open (not yet triaged) alerts in the Alerts console for its current method / seed / confirmation settings.
    None until the Alerts page has picked a method and seed (or when this run has no saved models)."""
    method, seed = st.session_state.get("ids_al_mth"), st.session_state.get("ids_al_seed")
    if method is None or seed is None:
        return None
    k, m, q = app_ids._alert_settings()
    try:
        al, _ = app_ids.build_alerts(run, seed, method, k, m, q)
    except Exception:
        return None
    tkey = f"{run}|{seed}|{method}|{k}/{m}/{q:g}"
    triage = st.session_state.get("ids_triage", {})
    n_open = sum(triage.get(f"{tkey}|{a}", "open") == "open" for a in al["id"])
    return {"open": int(n_open), "total": len(al), "method": method, "seed": seed}


BELL = ('<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px">'
        '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>')


def top_bar(run, meta, dataset, state, fl_round):
    """Server console bar: the status-bar fields plus FL round and the open-alert bell."""
    run_dir = os.path.join(app_ids.RUNS, run)
    color = {"done": "var(--ids-ok)", "running": "var(--ids-warn)", "preparing": "var(--ids-warn)",
             "failed": "var(--ids-bad)"}.get(state, "rgba(128,128,128,.8)")
    cc = (meta.get("config", {}) or {}).get("clients", {}) or {}
    al = open_alerts(run)
    if al is None:
        bell = f'<span title="Open the Alerts page to load the alert console">{BELL}–</span>'
    else:
        hot = "var(--ids-bad)" if al["open"] else "var(--ids-ok)"
        bell = (f'<span title="{al["open"]:,} open of {al["total"]:,} alerts · {app_ids.esc(app_ids.lab(al["method"]))}, '
                f'seed {al["seed"]} (Alerts page)" style="color:{hot};font-weight:700">{BELL}{al["open"]:,} open</span>')
    last = app_ids._last_update(run_dir, meta)
    items = [("dataset", app_ids.esc(dataset)), ("run", app_ids.esc(run)),
             ("state", f'<span class="ids-dot" style="background:{color}"></span>{app_ids.esc(state)}'),
             ("fl round", f"{fl_round} / {meta.get('rounds', '?')}"),
             ("model version", app_ids.esc(app_ids.model_version(run_dir, meta))),
             ("drones", f"{cc.get('num_clients', '?')} sites"),
             ("secure link", app_ids.esc(app_ids.secure_link_label(run_dir, meta) or "off")),
             ("alerts", bell), ("last update", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)))]
    body = "".join(f'<div><span class="k">{k}</span>{v}</div>' for k, v in items)
    st.markdown(f'<div class="ids-status">{body}</div>', unsafe_allow_html=True)
