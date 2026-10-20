"""Drone Sites panel of the dashboard (app.py): what each site / drone owner can see.

  Site map    the 10 sites around the aggregation server; readiness of each drone in the current round
              (live from the job's live_state.json while a run trains, else the last finished round)
  Drone page  one site: its local data and labels, its TRUE role (the owner knows it), its local training per round,
              and how the server sees it (the server-side trust score from app_server)

SITES is shared with the Secure Link visualisation (app_securelink.py): same names and grid coordinates everywhere.
Read-only: nothing is written to runs/, data/ or data_cache/.
"""
import glob
import math
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

import app_ids
import app_roles
import app_server

# 10 generic sites on a small local grid (km east / north of the aggregation server at 0, 0).
SITES = [
    {"cid": 0, "name": "Site A – North ridge", "x": 0.0, "y": 4.4},
    {"cid": 1, "name": "Site B – Pine hill", "x": 2.2, "y": 3.0},
    {"cid": 2, "name": "Site C – East quarry", "x": 4.3, "y": 1.4},
    {"cid": 3, "name": "Site D – River bend", "x": 3.6, "y": -1.2},
    {"cid": 4, "name": "Site E – Solar farm", "x": 2.5, "y": -3.5},
    {"cid": 5, "name": "Site F – South field", "x": 0.0, "y": -3.6},
    {"cid": 6, "name": "Site G – Old airstrip", "x": -2.6, "y": -3.6},
    {"cid": 7, "name": "Site H – West pass", "x": -3.5, "y": -1.1},
    {"cid": 8, "name": "Site I – Lake shore", "x": -4.3, "y": 1.4},
    {"cid": 9, "name": "Site J – Radio mast", "x": -2.2, "y": 3.1},
]
SERVER = {"name": "Aggregation server", "x": 0.0, "y": 0.0}


def site(cid):
    """Site of drone `cid`; runs with more than 10 drones get extra sites on an outer ring."""
    if cid < len(SITES):
        return SITES[cid]
    a = math.radians(90 - 360 * cid / 20)
    return {"cid": cid, "name": f"Site {chr(65 + cid) if cid < 26 else cid + 1}", "x": round(5.4 * math.cos(a), 1),
            "y": round(5.4 * math.sin(a), 1)}


def drone_name(cid):
    return f"Drone {cid + 1:02d}"


def site_label(cid):
    return f"{drone_name(cid)} · {site(cid)['name']}"


def n_drones(meta):
    return int(((meta.get("config", {}) or {}).get("clients", {}) or {}).get("num_clients", 10))


# ---------- readiness (live_state.json) ----------
# real mTLS adds: connecting (process started, TLS handshake / authentication not done), connected, refused
LIVE_STATES = ("training", "ready", "sent", "received", "connecting", "connected", "refused")
FADED = ("training", "connecting", "refused")


def live_jobs(run_dir):
    """Job folders of this run that have a live_state.json (written while training with the secure link on),
    newest first: [(method, seed, path)]. A run-level live_state.json counts as one job too."""
    paths = glob.glob(os.path.join(run_dir, "*", "seed_*", "live_state.json"))
    paths += [p for p in [os.path.join(run_dir, "live_state.json")] if os.path.exists(p)]
    out = []
    for p in sorted(paths, key=os.path.getmtime, reverse=True):
        rel = os.path.relpath(p, run_dir).split(os.sep)
        method, seed = (rel[0], rel[1].replace("seed_", "")) if len(rel) == 3 else ("", "")
        out.append((method, seed, p))
    return out


def read_live_state(path, n):
    """{cid: {"state", "t", "progress"?}} for drones 0..n-1 plus round / phase; missing drones count as training."""
    raw = app_ids.load_json(path, {}) or {}
    drones = {cid: {"state": "training"} for cid in range(n)}
    for d in raw.get("drones", []) or []:
        try:
            cid = int(d.get("cid"))
        except (TypeError, ValueError):
            continue
        if 0 <= cid < n:
            drones[cid] = {"state": d.get("state") if d.get("state") in LIVE_STATES else "training", "t": d.get("t"),
                           "progress": d.get("progress")}
    age = time.time() - os.path.getmtime(path) if os.path.exists(path) else None
    return {"round": raw.get("round"), "phase": raw.get("phase") or "", "drones": drones, "age": age}


def _palette():
    if st.session_state.get("ui_dark"):
        return {"bg": "#0e1217", "grid": "rgba(213,221,230,.055)", "axis": "rgba(213,221,230,.16)", "text": "#d5dde6",
                "muted": "#8b97a6", "accent": "#4a8cf7", "ok": "#4fbf63", "link": "rgba(150,165,185,.30)",
                "panel": "rgba(110,135,170,.10)", "border": "rgba(150,165,185,.28)", "check_text": "#0e1217",
                "bad": "#ff6b6b"}
    return {"bg": "#ffffff", "grid": "rgba(49,51,63,.055)", "axis": "rgba(49,51,63,.16)", "text": "#31333f",
            "muted": "#6b7280", "accent": "#2563eb", "ok": "#3f9f4f", "link": "rgba(100,110,125,.30)",
            "panel": "rgba(128,128,128,.07)", "border": "rgba(128,128,128,.30)", "check_text": "#ffffff",
            "bad": "#bd271e"}


def _drone_icon(stroke):
    """Quadcopter line icon centred on (0, 0): body, four arms, four rotors with a blade line, camera dot."""
    rotors = "".join(f'<circle cx="{x}" cy="{y}" r="7.5" class="rot"/><line x1="{x - 5}" y1="{y}" x2="{x + 5}" y2="{y}" '
                     f'class="blade"/>' for x, y in ((-13, -13), (13, -13), (-13, 13), (13, 13)))
    return (f'<g fill="none" stroke="{stroke}" stroke-width="1.6" stroke-linecap="round">'
            '<line x1="-4" y1="-4" x2="-10" y2="-10"/><line x1="4" y1="-4" x2="10" y2="-10"/>'
            '<line x1="-4" y1="4" x2="-10" y2="10"/><line x1="4" y1="4" x2="10" y2="10"/>'
            f'{rotors}<rect x="-6" y="-6" width="12" height="12" rx="3.5"/>'
            f'<circle cx="0" cy="0" r="1.6" fill="{stroke}" stroke="none"/></g>')


def _server_icon(p):
    slots = "".join(f'<rect x="-22" y="{y}" width="44" height="11" rx="2"/><circle cx="-15" cy="{y + 5.5}" r="1.6" '
                    f'fill="{p["ok"]}" stroke="none"/><line x1="-6" y1="{y + 5.5}" x2="14" y2="{y + 5.5}" '
                    f'stroke-dasharray="2 2.5"/>' for y in (-19, -5.5, 8))
    return (f'<g fill="none" stroke="{p["text"]}" stroke-width="1.5" stroke-linecap="round">'
            f'<rect x="-34" y="-32" width="68" height="64" rx="8" fill="{p["bg"]}" stroke="none"/>'
            f'<rect x="-28" y="-26" width="56" height="52" rx="5" fill="{p["panel"]}" stroke="{p["border"]}"/>{slots}</g>')


def map_html(n, states, prev, header, sub, roles=None, mtls=False):
    """Hand-built SVG site map. states: {cid: {"state", "progress"?}}; prev: {cid: opacity} from the previous draw,
    so each drone fades smoothly from its old to its new opacity. roles: {cid: role} colours the drone's name and id
    (poisoned red, rare green; the site owner knows its role). mtls: real mTLS run, the legend adds "Connecting…"
    (drone process started, TLS handshake not done yet: faded, dashed link)."""
    p = _palette()
    W, H, S = 900, 668, 54          # canvas and px per km
    cx, cy = W / 2, 338
    X = lambda x: cx + x * S
    Y = lambda y: cy - y * S
    grid = "".join(f'<line x1="{X(k)}" y1="{Y(-5)}" x2="{X(k)}" y2="{Y(5)}"/>' for k in range(-7, 8)) + \
        "".join(f'<line x1="{X(-7)}" y1="{Y(k)}" x2="{X(7)}" y2="{Y(k)}"/>' for k in range(-5, 6))
    ticks = "".join(f'<text x="{X(k)}" y="{Y(-5) + 14}" text-anchor="middle">{k:+d}</text>'.replace('+0<', '0<') for k in range(-6, 7, 2)) + \
        "".join(f'<text x="{X(-7) - 6}" y="{Y(k) + 3}" text-anchor="end">{k:+d}</text>'.replace('+0<', '0<') for k in range(-4, 5, 2))
    links, nodes, keyframes = [], [], []
    ready = received = 0
    for cid in range(n):
        s = site(cid)
        st_ = states.get(cid, {}).get("state", "training")
        ready += st_ not in FADED
        received += st_ == "received"
        x, y = X(s["x"]), Y(s["y"])
        o1 = 0.4 if st_ in FADED else 1.0
        o0 = prev.get(cid, o1)
        stroke = p["muted"] if st_ in FADED else (p["ok"] if st_ == "received" else p["accent"])
        lk = {"training": f'stroke="{p["link"]}" stroke-dasharray="3 5"', "ready": f'stroke="{p["link"]}"',
              "connecting": f'stroke="{p["muted"]}" stroke-opacity=".7" stroke-dasharray="8 6"',  # dashed: no session yet
              "connected": f'stroke="{p["link"]}"', "refused": f'stroke="{p["link"]}" stroke-opacity=".35" stroke-dasharray="1 7"',
              "sent": f'stroke="{p["accent"]}" stroke-opacity=".55"', "received": f'stroke="{p["ok"]}" stroke-opacity=".45"'}[st_]
        links.append(f'<line id="l{cid}" x1="{x}" y1="{y}" x2="{cx}" y2="{cy}" {lk} stroke-width="1.2"/>')
        if st_ == "sent":  # update in transit: a small capsule travelling to the server
            links.append(f'<rect x="-7" y="-3" width="14" height="6" rx="3" fill="{p["accent"]}"><animateMotion dur="1.6s" '
                         f'repeatCount="indefinite" rotate="auto" path="M{x},{y} L{cx},{cy}"/></rect>')
        ring = ""
        if st_ == "training":
            prog = states.get(cid, {}).get("progress")
            if isinstance(prog, (int, float)) and 0 <= prog <= 1:
                circ = 2 * 3.14159 * 26
                ring = (f'<circle r="26" fill="none" stroke="{p["axis"]}" stroke-width="2"/>'
                        f'<circle r="26" fill="none" stroke="{p["accent"]}" stroke-width="2" stroke-linecap="round" '
                        f'stroke-dasharray="{circ * prog:.1f} {circ:.1f}" transform="rotate(-90)"/>')
            else:
                ring = (f'<circle r="26" fill="none" stroke="{p["accent"]}" stroke-width="1.6" stroke-linecap="round" '
                        f'stroke-dasharray="22 142" class="spin"/>')
        if st_ == "connecting":  # dashed ring: waiting for the TLS handshake
            ring = (f'<circle r="26" fill="none" stroke="{p["muted"]}" stroke-width="1.4" stroke-dasharray="4 4" '
                    f'class="spin"/>')
        check = ""
        if st_ == "received":
            check = (f'<g transform="translate(19,-19)"><circle r="7" fill="{p["ok"]}"/><path d="M-3.2,0.2 L-0.8,2.6 L3.4,-2.4" '
                     f'fill="none" stroke="{p["check_text"]}" stroke-width="1.8" stroke-linecap="round" '
                     f'stroke-linejoin="round"/></g>')
        label = {"training": "Training…", "connecting": "Connecting…", "connected": "Connected",
                 "refused": "Refused"}.get(st_, "Ready")
        lab_col = p["bad"] if st_ == "refused" else (p["muted"] if st_ in FADED else p["text"])
        name = s["name"].split("–")[-1].strip()
        rc = app_roles.color((roles or {}).get(cid))
        fill = f' style="fill:{rc}"' if rc else ""  # inline style: the .nm / .id class fill would win over fill="..."
        keyframes.append(f"@keyframes f{cid}{{from{{opacity:{o0}}}to{{opacity:{o1}}}}}")
        tip = f"{s['name']} · {drone_name(cid)} · grid ({s['x']:+.1f}, {s['y']:+.1f}) km · {st_}"
        nodes.append(
            f'<g class="node" transform="translate({x},{y})" style="opacity:{o1};animation:f{cid} .9s ease both">'
            f'<title>{app_ids.esc(tip)}</title>'
            f'<circle r="30" fill="{p["bg"]}" stroke="none"/>{ring}{_drone_icon(stroke)}{check}'
            f'<text y="44" text-anchor="middle" class="nm"{fill}>{app_ids.esc(name)}</text>'
            f'<text y="57" text-anchor="middle" class="id"{fill}>{drone_name(cid).upper()} · '
            f'<tspan fill="{lab_col}" class="st">{label}</tspan></text></g>')
    legend_items = [("Training…", p["muted"], 0.4, ""), ("Ready", p["accent"], 1.0, ""),
                    ("Update sent", p["accent"], 1.0, "sent"), ("Received by server", p["ok"], 1.0, "check")]
    if mtls:
        legend_items.insert(0, ("Connecting…", p["muted"], 0.4, "dash"))
    # mTLS runs have one more entry: spaced by label length so all five fit left of the grid note
    step = [125, 105, 90, 125, 160] if mtls else [150] * len(legend_items)
    legend = []
    for i, (txt, col, op, extra) in enumerate(legend_items):
        lx = 22 + sum(step[:i])
        mark = f'<g transform="translate({lx + 10},{H - 22}) scale(.55)" style="opacity:{op}">{_drone_icon(col)}</g>'
        if extra == "sent":
            mark += f'<rect x="{lx + 24}" y="{H - 25}" width="12" height="6" rx="3" fill="{col}"/>'
        if extra == "dash":
            mark += (f'<line x1="{lx + 22}" y1="{H - 22}" x2="{lx + 36}" y2="{H - 22}" stroke="{col}" stroke-width="1.4" '
                     f'stroke-dasharray="4 3"/>')
        if extra == "check":
            mark += (f'<g transform="translate({lx + 21},{H - 32}) scale(.7)"><circle r="7" fill="{col}"/>'
                     f'<path d="M-3.2,0.2 L-0.8,2.6 L3.4,-2.4" fill="none" stroke="{p["check_text"]}" stroke-width="1.8" '
                     f'stroke-linecap="round"/></g>')
        legend.append(f'{mark}<text x="{lx + 40}" y="{H - 18}" class="lg">{txt}</text>')
    css = f"""
    html,body{{margin:0;background:{p['bg']};font-family:"Source Sans Pro","Source Sans 3",-apple-system,BlinkMacSystemFont,
    "Segoe UI",sans-serif}}
    svg{{width:100%;height:auto;max-height:668px;display:block}}
    .grid line{{stroke:{p['grid']};stroke-width:1}} .tick text{{fill:{p['muted']};font:10px ui-monospace,Menlo,monospace}}
    .nm{{fill:{p['text']};font-size:12.5px;font-weight:600}}
    .id{{fill:{p['muted']};font:10px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.04em}}
    .st{{font-family:"Source Sans Pro",-apple-system,sans-serif;font-size:10.5px;font-weight:600;letter-spacing:0}}
    .lg{{fill:{p['muted']};font-size:11.5px}}
    .hd{{fill:{p['text']};font-size:15px;font-weight:600}} .sb{{fill:{p['muted']};font:11px ui-monospace,Menlo,monospace}}
    .node{{transition:opacity .8s ease}}
    .spin{{transform-origin:0 0;animation:spin 1.6s linear infinite}}
    @keyframes spin{{to{{transform:rotate(360deg)}}}}
    {''.join(keyframes)}
    """
    count = (f'<text x="{W - 20}" y="30" text-anchor="end" class="hd">{ready}/{n} ready</text>'
             f'<text x="{W - 20}" y="47" text-anchor="end" class="sb">{received}/{n} updates received</text>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Drone site map">
<g class="grid">{grid}</g><g class="tick">{ticks}</g>
<text x="{W - 20}" y="{H - 18}" text-anchor="end" class="sb">1 km grid · server at 0, 0</text>
<text x="20" y="30" class="hd">{app_ids.esc(header)}</text><text x="20" y="47" class="sb">{app_ids.esc(sub)}</text>{count}
{''.join(links)}
<g transform="translate({cx},{cy})">{_server_icon(p)}
<text y="44" text-anchor="middle" class="nm">{SERVER['name']}</text>
<text y="57" text-anchor="middle" class="id">FILTER · VOTE · DQN WEIGHT</text></g>
{''.join(nodes)}
{''.join(legend)}
</svg></body></html>"""


def _draw(n, states, header, sub, roles=None, mtls=False):
    prev = st.session_state.get("site_map_prev", {})
    html_doc = map_html(n, states, prev, header, sub, roles, mtls)
    if hasattr(st, "iframe"):  # st.iframe replaces the deprecated components.v1.html (same srcdoc iframe)
        st.iframe(html_doc, height="content", alt="Drone site map: ten sites around the aggregation server")
    else:
        import streamlit.components.v1 as components
        components.html(html_doc, height=680)
    st.session_state["site_map_prev"] = {c: (0.4 if states.get(c, {}).get("state") in FADED else 1.0)
                                         for c in range(n)}


def _live_map(path, n, label, roles=None, mtls=False):
    ls = read_live_state(path, n)
    rnd = ls["round"]
    head = f"Round {rnd + 1} · {ls['phase'] or 'in progress'}" if isinstance(rnd, int) else "Current round"
    _draw(n, ls["drones"], head, f"live · {label}", roles, mtls)
    if ls["age"] is not None:  # outside the drawing, so an unchanged map is not redrawn every poll
        st.caption(f"Readiness file updated {ls['age']:.0f} s ago · refreshes every 1.5 s")


def tab_site_map(run_dir, meta, live, fl_round=None):
    app_ids.note("The ten sites around the aggregation server. Each site labels its own drone's traffic and trains the "
                 "local model; only model updates travel (encrypted) to the server. Faded drones are still training this "
                 "round; drones at full strength have finished; a green check means the server has received the update.")
    n = n_drones(meta)
    parts = (meta.get("partitions") or {})
    p = parts.get(str((meta.get("seeds") or [None])[0])) or {}
    by_cid = {c["cid"]: c for c in p.get("clients", [])}
    roles = {c: x.get("role") for c, x in by_cid.items()}
    mtls = (((meta.get("config") or {}).get("security") or {}).get("transport") == "mtls")
    jobs = live_jobs(run_dir) if live else []
    if jobs:
        labels = [f"{app_ids.lab(m)} · seed {s}" if m else "this run" for m, s, _ in jobs]
        i = 0
        if len(jobs) > 1:
            i = st.selectbox("Training job", range(len(jobs)), format_func=labels.__getitem__, key="site_job",
                             help="Jobs that are training now with the secure link on (newest first).")
        st.fragment(run_every=1.5)(_live_map)(jobs[i][2], n, labels[i], roles, mtls)
    else:
        rounds = fl_round or meta.get("rounds", "?")
        why = "training without the secure link (no live readiness file)" if live else "finished run"
        _draw(n, {c: {"state": "received"} for c in range(n)}, f"Round {rounds} · last finished round",
              f"{why} · every drone ready, every update received", roles, mtls)
    st.dataframe(app_roles.rows(pd.DataFrame([{"Site": site(c)["name"], "Drone": drone_name(c),
                                               "Grid east (km)": site(c)["x"], "Grid north (km)": site(c)["y"],
                                               "Distance to server (km)": round((site(c)["x"] ** 2 + site(c)["y"] ** 2) ** 0.5, 1),
                                               "Local training rows": by_cid.get(c, {}).get("n_train")} for c in range(n)]),
                                [roles.get(c) for c in range(n)], ["Site", "Drone"]),
                 hide_index=True, width="stretch")
    app_ids.note("Grid coordinates are a small local grid in km around the server (illustrative site layout). "
                 "Local training rows: first seed of this run.")


# ---------- per-drone page ----------
def tab_drone(run_dir, meta, methods, seeds):
    app_ids.note("One site's own view: its data and labels, its true role (the site owner knows it; the server never "
                 "does), what happened to its model each round, and the trust the server has in it.")
    n = n_drones(meta)
    c = st.columns([2, 3, 1])
    cid = c[0].selectbox("Drone / site", list(range(n)), format_func=site_label, key="site_cid")
    jm = [m for m in methods if not m.startswith("fedavg_clean")] or methods
    with c[1]:
        method = st.radio("Method", jm, horizontal=True, format_func=app_ids.lab, key="site_m",
                          index=next((i for i, m in enumerate(jm) if m.startswith("jiang_dqn")), 0))
    with c[2]:
        seed = st.radio("Seed", seeds, horizontal=True, key="site_s")
    part = ((meta.get("partitions") or {}).get(str(seed)) or {}).get("clients", [])
    me = next((x for x in part if x["cid"] == cid), None)
    s = site(cid)
    role = (me or {}).get("role")
    role_name = app_server.ROLE_NAMES.get(role, role or "–")
    role_col = app_server.ROLE_COLORS.get(role_name, "#607d8b")
    st.markdown(f'<div class="ids-banner" style="border-left-color:{role_col}"><b>{app_roles.span(s["name"], role)}</b> · '
                f'{app_roles.span(drone_name(cid), role)} · grid ({s["x"]:+.1f}, {s["y"]:+.1f}) km &nbsp;·&nbsp; true role: '
                f'<b style="color:{role_col}">{app_ids.esc(role_name)}</b> <span style="opacity:.65">(known to this site '
                f'only; the server never sees it)</span></div>', unsafe_allow_html=True)

    cf = app_server.client_frame(run_dir, method, seed)
    mine = cf[cf["cid"] == cid] if not cf.empty else cf
    tr = app_server.trust_frame(run_dir, method, seed)
    t_me = tr[tr["cid"] == cid] if not tr.empty else tr
    k = st.columns(5)
    k[0].metric("Local training rows", f"{me['n_train']:,}" if me else "–")
    k[1].metric("Voting set rows", f"{me['n_vote_set']:,}" if me and me.get("n_vote_set") is not None else "–")
    k[2].metric("Classes held", sum(v > 0 for v in me["counts"].values()) if me else "–")
    if not t_me.empty:
        now = float(t_me["trust"].iloc[-1])
        k[3].metric("Server trust now", f"{now:.0f} / 100", help=app_server.TRUST_HELP)
        k[4].metric("Trust band", app_server.band(now))
    if role == "poisoned":
        flip = ((meta.get("config", {}) or {}).get("clients", {}) or {}).get("flip_mode", "")
        st.caption(f"This site's labelling is compromised: its labels are flipped ({flip}) before local training. "
                   "The class counts below are the true labels.")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("##### Local data and labels")
        if me:
            df = pd.DataFrame([{"Class": k_, "Rows": v} for k_, v in me["counts"].items()])
            rare = meta.get("rare_class")
            base = alt.Chart(df).encode(y=alt.Y("Class:N", sort=list(me["counts"]), title=None))
            bars = base.mark_bar(height=14).encode(
                x=alt.X("Rows:Q", title="Rows (true labels)"),
                color=alt.condition(alt.datum.Class == rare, alt.value("#2e7d32"), alt.value("#607d8b")),
                tooltip=["Class", alt.Tooltip("Rows:Q", format=",")])
            txt = base.mark_text(align="left", dx=4, fontSize=11, color=app_ids.ink()).encode(
                x="Rows:Q", text=alt.Text("Rows:Q", format=","))
            st.altair_chart((bars + txt).properties(height=alt.Step(22)), width="stretch")
        else:
            st.info("No partition recorded for this seed.")
    with c2:
        st.markdown("##### How the server sees this drone")
        if t_me.empty:
            st.info("No rounds yet for this method and seed.")
        else:
            st.altair_chart(app_server.trust_chart(t_me.assign(drone=drone_name(cid)), meta, single=True),
                            width="stretch")
            app_ids.note("The server's trust score for this drone over the rounds (same formula as Aggregation Server → "
                         "Trust scores; it uses only what the server observes, never the true role).")

    st.markdown("##### Local training, round by round")
    if mine.empty:
        st.info("No rounds yet.")
        return
    vs = meta.get("vote_start_round", 0)
    d = mine.assign(Round=mine["round"] + 1,
                    Decision=[app_server.status_label(x, r, vs) for x, r in zip(mine["status"], mine["round"])])
    present = [lbl for lbl, _ in app_server.STATUS.values() if lbl in set(d["Decision"])]
    scale = alt.Scale(domain=present, range=[dict(app_server.STATUS.values())[x] for x in present])
    strip = alt.Chart(d).mark_rect().encode(
        x=alt.X("Round:Q", title=None, bin=alt.Bin(maxbins=min(len(d), 400)), axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("Decision:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Round", "Decision"]).properties(height=26)
    w = alt.Chart(d).mark_line(strokeWidth=1.6, color="#1565c0").encode(
        x=alt.X("Round:Q", title="FL round"), y=alt.Y("weight:Q", title="Weight in the global model",
                                                      axis=alt.Axis(format="%")),
        tooltip=["Round", alt.Tooltip("weight:Q", format=".1%"), "Decision"]).properties(height=150)
    charts = [strip, w]
    if d["score"].notna().any():
        charts.append(alt.Chart(d).mark_line(strokeWidth=1.4, color="#8e24aa").encode(
            x=alt.X("Round:Q", title="FL round"), y=alt.Y("score:Q", title="Filter distance", scale=alt.Scale(zero=False)),
            tooltip=["Round", alt.Tooltip("score:Q", format=".2f")]).properties(height=120))
    st.altair_chart(alt.vconcat(*charts).resolve_scale(x="shared"), width="stretch")
    app_ids.note("Top strip: what the server decided about this site's update each round. Weight: its share of the "
                 "global model. Filter distance: sum of distances to the other updates (lower = more central).")
    with st.expander("Round table"):
        tbl = d.sort_values("round", ascending=False)
        st.dataframe(pd.DataFrame({
            "Round": tbl["Round"], "Decision": tbl["Decision"], "Weight": tbl["weight"], "Filter distance": tbl["score"],
            "Votes yes": [f"{int(a)}/{int(b)}" if pd.notna(b) else "–" for a, b in zip(tbl["votes_yes"], tbl["votes_total"])],
            "DQN weight": tbl["dqn_weight"]}).style.format({"Weight": "{:.2%}", "Filter distance": "{:.3f}",
                                                             "DQN weight": "{:.2f}"}, na_rep="–"),
            hide_index=True, width="stretch", height=320)
