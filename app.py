"""FL DoS Swarm dashboard: launch experiments and check them against Jiang et al. (2024).

Every chart answers one question from the paper:
  Overview   Table 2  - does Jiang keep the rare class (like FedAvg) AND resist poisoning (like Multi-Krum)?
             Fig. 6   - how much say does each drone type get in the global model?
             Votes    - poisoned voted out, rare and regular drones voted back in?
             Attack   - how does each method hold up as the number of poisoned drones grows?
  Curves     Fig. 5   - training progress, smoothed, with the spread across seeds
  Decisions           - per drone: how often selected / voted back in / voted out
  Results             - numbers, and confusion matrices averaged over seeds (De-authentication row highlighted)
"""
import json
import os
import signal
import subprocess
import sys
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import app_roles  # role colours: poisoned drones red, the rare drone green
import app_ids  # IDS-console additions (Test, Live, Alerts, Verification tabs; status bar; CSS)
import app_server  # Aggregation Server panel: top bar, trust scores, audit log
import app_sites  # Drone Sites panel: site map (readiness), per-drone page; SITES shared with the Secure Link view
import app_mtls  # real mTLS link: connection table, wire traffic, certificates (project PKI) with revoke / restore

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
DEFAULT_CFG = os.path.join(ROOT, "configs", "default.json")
# Training from the dashboard starts from the settings each dataset's final run actually used (configs/recommended,
# never written by the UI); see configs/recommended/README.md.
REC = os.path.join(ROOT, "configs", "recommended")
DATASET_CFG = {"UAV-NIDD": os.path.join(REC, "uav_nidd.json"), "ISOT Drone": os.path.join(REC, "isot.json"),
               "ISOT Drone · 72 inputs": os.path.join(REC, "isot72.json"),
               "UAV-NIDD · plain": os.path.join(REC, "nidd_plain.json"),
               "UAV-NIDD · CPU-only": os.path.join(REC, "uav_nidd.json")}


def chosen_cfg():
    """Config of the dataset selected in the sidebar switch (training and pre-training on command use it)."""
    ds = st.session_state.get("dataset_sel", "UAV-NIDD")
    return ds, DATASET_CFG.get(ds, DEFAULT_CFG)

METHOD_LABELS = {"jiang_dqn": "Jiang + DQN", "jiang_dqn_pre": "Jiang + pre-trained DQN",
                 "jiang_dqn_prexv": "Jiang + DQN pre-trained across versions",
                 "jiang_fixed": "Jiang, fixed weight", "fedavg": "FedAvg", "multi_krum": "Multi-Krum",
                 "fedwgca": "FedWGCA (robust side)", "fed_grab": "Fed-GraB (rare side)",
                 "fedavg_clean": "FedAvg, no attackers (reference)"}
METHOD_ORDER = list(METHOD_LABELS)
METHOD_COLORS = {"jiang_dqn": "#1565c0", "jiang_dqn_pre": "#00897b", "jiang_dqn_prexv": "#004d40",
                 "jiang_fixed": "#90caf9",
                 "fedavg": "#ef6c00", "multi_krum": "#8e24aa", "fedavg_clean": "#9e9e9e",
                 "fedwgca": "#c2185b", "fed_grab": "#f9a825"}
HEADLINE = ["fedavg", "multi_krum", "jiang_dqn"]

# Display name per decision, in the paper's terms. "frozen" = dropped while the voting layer is still frozen.
STATUS = {"accepted": ("Selected by filter", "#2e7d32"), "readmitted": ("Voted back in", "#1e88e5"),
          "vetoed": ("Voted in, DQN weight 0", "#8e24aa"), "rejected": ("Voted out", "#c62828"),
          "flagged": ("Dropped, vote next round", "#f2a900"), "frozen": ("Dropped, voting not started", "#bdbdbd"),
          "averaged": ("Averaged (FedAvg)", "#757575"), "excluded": ("Excluded (oracle)", "#424242")}
STATUS_HELP = ("**Selected by filter**: among the K most central models (the DQN may also set its weight) · "
               "**Dropped, vote next round**: outside the filter cluster, every drone votes on it after `vote_lag` "
               "rounds (0 = same round) · **Voted back in**: the vote readmitted the dropped model and the DQN weighted it · **Voted out**: the vote rejected it · **Voted in, DQN "
               "weight 0**: readmitted but the DQN gave it no weight · **Dropped, voting not started**: the voting "
               "layer is frozen for the first rounds (paper Sec. 3)")
ROLE_NAMES = {"general": "Regular", "rare": "Rare (De-auth)", "poisoned": "Poisoned"}
ROLE_COLORS = {"Regular": "#607d8b", "Rare (De-auth)": "#2e7d32", "Poisoned": "#c62828"}
SHORT = {"De-authentication": "De-auth"}


def set_dataset(meta):
    """Point dataset-specific wording at the selected run (rare class, dataset name); UAV-NIDD keeps the old text."""
    rare = (meta.get("config", {}).get("data", {}) or {}).get("rare_class") or meta.get("rare_class") or "De-authentication"
    label = f"Rare ({SHORT.get(rare, rare)})"
    ROLE_NAMES["rare"] = label
    for k in [k for k in ROLE_COLORS if k.startswith("Rare (")]:
        del ROLE_COLORS[k]
    ROLE_COLORS[label] = app_roles.color("rare")  # lighter red / green in dark mode
    ROLE_COLORS["Poisoned"] = app_roles.color("poisoned")


def dataset_name(meta):
    if meta.get("dataset_label"):  # set on runs that get their own entry (e.g. the CPU-only UAV-NIDD re-run)
        return meta["dataset_label"]
    data = (meta.get("config", {}) or {}).get("data", {}) or {}
    if "isot" in str(data.get("csv_path", "")).lower():
        # ISOT with the per-packet protocol fields (72 inputs) is shown as its own entry next to the 29-input ISOT
        return "ISOT Drone · 72 inputs" if data.get("proto_path") else "ISOT Drone"
    # UAV-NIDD trained without class weights (plain loss) is shown as its own entry next to the class-weighted one
    plain = ((meta.get("config", {}) or {}).get("training", {}) or {}).get("class_weighting") == "none"
    return "UAV-NIDD · plain" if plain else "UAV-NIDD"
ROLE_ICONS = {"general": "", "rare": "rare", "poisoned": "poisoned"}

st.set_page_config(page_title="FL DoS Swarm", page_icon=":material/shield:", layout="wide")


# ---------- files ----------
def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def read_rounds(run_dir, method, seed):
    rows = []
    try:
        with open(os.path.join(run_dir, method, f"seed_{seed}", "rounds.jsonl")) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # half-written last line while training
                    break
    except OSError:
        pass
    return rows


def summaries(run_dir, methods, seeds):
    out = {}
    for m in methods:
        for s in seeds:
            v = load_json(os.path.join(run_dir, m, f"seed_{s}", "summary.json"))
            if v:
                out[(m, s)] = v
    return out


def list_runs():
    if not os.path.isdir(RUNS):
        return []
    runs = [d for d in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, d, "meta.json"))]
    return sorted(runs, key=lambda d: os.path.getmtime(os.path.join(RUNS, d, "meta.json")), reverse=True)


def pid_alive(pid):
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:  # a run this server started that has exited: reap it
            return False
    except (ChildProcessError, OSError, TypeError):
        pass  # not our child (e.g. started from a terminal): fall through to the plain check
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def run_state(meta):
    s = meta.get("status", "?")
    return "stopped" if s in ("preparing", "running") and not pid_alive(meta.get("pid")) else s


def drone(cid):
    return f"Drone {cid + 1:02d}"


def ordered(methods):
    return sorted(methods, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)


def status_key(d, rnd, vote_start):
    return "frozen" if d["status"] == "flagged" and rnd < vote_start else d["status"]


def method_scale(methods):
    return alt.Scale(domain=[METHOD_LABELS[m] for m in methods], range=[METHOD_COLORS[m] for m in methods])


def spawn(args, log_path):
    log = open(log_path, "w")
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1"}
    return subprocess.Popen([sys.executable, os.path.join(ROOT, "run_experiment.py"), *args], cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True, env=env)


# ---------- sidebar ----------
@st.cache_data(show_spinner=False, max_entries=16)
def dataset_card(run, mtime, dark):
    """Hover card for a dataset button (from its final run): setup, then each class's rows on the drones (seed 1, all
    drones, training data) grouped DoS types / Normal / other attacks, with counts, shares and totals."""
    m = load_json(os.path.join(RUNS, run, "meta.json"), {})
    c = m.get("config", {}) or {}
    d, tr = c.get("data", {}) or {}, c.get("training", {}) or {}
    part = (m.get("partitions") or {}).get("1") or next(iter((m.get("partitions") or {}).values()), {})
    cnt = {}
    for cl in part.get("clients", []):
        for k, v in cl.get("counts", {}).items():
            cnt[k] = cnt.get(k, 0) + v
    if not cnt:
        return ""
    total = sum(cnt.values())
    normal = d.get("normal_class", "Normal")
    dos = [k for k in (d.get("dos_classes") or []) if k in cnt]
    other = [k for k in cnt if k != normal and k not in dos]
    rare = d.get("rare_class") or m.get("rare_class")
    cw = tr.get("class_weighting", "none")
    loss = "Plain loss (no class weights)" if cw == "none" else f"Class-weighted ({cw.replace('_', ' ')})"
    inputs = "72 inputs (size/timing + protocol fields)" if d.get("proto_path") else "29 inputs (size/timing)"
    gpu = m.get("gpu_seeds") or []
    device = f"CPU + GPU (seeds {', '.join(map(str, gpu))} on GPU)" if gpu else "CPU only"
    esc = lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;")
    rare_col = "#66bb6a" if dark else "#2e7d32"

    def row(name, n, cls=""):
        tag = f' <span style="color:{rare_col};font-weight:600">rare</span>' if name == rare else ""
        return (f'<tr class="{cls}"><td class="k">{esc(name)}{tag}</td><td class="v">{n:,}</td>'
                f'<td class="v">{n / total:.2%}</td></tr>')

    rows = [f'<tr class="grp"><td colspan="3">DoS types</td></tr>'] + [row(k, cnt[k]) for k in dos]
    rows += [row("All DoS", sum(cnt[k] for k in dos), "sub")]
    rows += [f'<tr class="grp"><td colspan="3">Normal</td></tr>', row(normal, cnt.get(normal, 0))]
    if other:
        rows += [f'<tr class="grp"><td colspan="3">Other attacks</td></tr>'] + [row(k, cnt[k]) for k in other]
    rows += [f'<tr class="tot"><td class="k">Total</td><td class="v">{total:,}</td><td class="v">100%</td></tr>']
    head = (f'<div class="h">{esc(dataset_name(m))}</div>'
            f'<table class="meta"><tr><td class="k">Loss</td><td>{esc(loss)}</td></tr>'
            f'<tr><td class="k">Inputs</td><td>{esc(inputs)}</td></tr>'
            f'<tr><td class="k">Device</td><td>{esc(device)}</td></tr>'
            f'<tr><td class="k">Run</td><td>{esc(run)} · {len(m.get("methods", []))} methods · '
            f'{m.get("rounds", tr.get("rounds", "?"))} rounds · seeds {", ".join(map(str, m.get("seeds", [])))}</td></tr></table>')
    foot = (f'<div class="f">Training rows held by the {len(part.get("clients", []))} drones (seed 1). '
            f'Validation {part.get("n_val", 0):,} · test {part.get("n_test", 0):,} rows'
            + (" (test stream includes replayed Normal)" if cnt.get(normal, 0) / total < 0.05 else "") + ".</div>")
    return head + '<table class="cls"><tr class="hd"><td></td><td class="v">Rows</td><td class="v">Share</td></tr>' + \
        "".join(rows) + "</table>" + foot


def dataset_hover(finals, have):
    """Hover cards on the sidebar's dataset buttons (styled like the charts' tooltips)."""
    dark = bool(st.session_state.get("ui_dark"))
    cards = {}
    for ds in have:
        run = finals.get(ds)
        mp = os.path.join(RUNS, run or "", "meta.json")
        if run and os.path.exists(mp):
            cards[ds] = dataset_card(run, os.path.getmtime(mp), dark)
    bg, fg, bd, mu = (("#2b2f36", "#e6eaf1", "#4a505b", "#9aa4b2") if dark else ("#ffffff", "#1f2933", "#d9d9d9", "#6b7280"))
    css = (f"#ds-tip{{position:fixed;z-index:2147483647;display:none;pointer-events:none;max-width:420px;background:{bg};"
           f"color:{fg};border:1px solid {bd};border-radius:4px;box-shadow:2px 2px 6px rgba(0,0,0,.18);padding:8px 10px;"
           f"font:12px/1.35 'Source Sans Pro',sans-serif}}#ds-tip .h{{font-weight:700;font-size:13px;margin-bottom:4px}}"
           f"#ds-tip table{{border-collapse:collapse;width:100%}}#ds-tip td{{padding:1px 4px;vertical-align:top}}"
           f"#ds-tip td.k{{font-weight:600;color:{mu};white-space:nowrap}}#ds-tip td.v{{text-align:right;"
           f"font-variant-numeric:tabular-nums;white-space:nowrap}}#ds-tip table.meta{{margin-bottom:6px}}"
           f"#ds-tip tr.grp td{{padding-top:5px;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}"
           f"#ds-tip tr.hd td{{color:{mu};font-size:11px;border-bottom:1px solid {bd}}}"
           f"#ds-tip tr.sub td{{border-top:1px dashed {bd};font-style:italic}}#ds-tip tr.tot td{{border-top:1px solid {bd};"
           f"font-weight:700}}#ds-tip tr.tot td.k{{color:{fg}}}#ds-tip .f{{margin-top:6px;color:{mu};font-size:11px}}")
    # a zero-height frame adds the style + listeners to the dashboard page itself (same origin), so the cards can
    # float over the sidebar like the charts' tooltips
    import streamlit.components.v1 as components
    components.html(f"""<script>(function(){{
  const doc = window.parent.document, win = window.parent;
  win.__dsCards = {json.dumps(cards)};
  let st = doc.getElementById('ds-tip-css');
  if (!st) {{ st = doc.createElement('style'); st.id = 'ds-tip-css'; doc.head.appendChild(st); }}
  st.textContent = {json.dumps(css)};
  let tip = doc.getElementById('ds-tip');
  if (!tip) {{ tip = doc.createElement('div'); tip.id = 'ds-tip'; doc.body.appendChild(tip); }}
  function move(e) {{
    const w = tip.offsetWidth, h = tip.offsetHeight;
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    let x = Math.max(e.clientX + 16, sb ? sb.getBoundingClientRect().right + 10 : 0), y = e.clientY - 20;
    if (x + w > win.innerWidth - 8) x = e.clientX - w - 16;
    if (y + h > win.innerHeight - 8) y = Math.max(8, win.innerHeight - h - 8);
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }}
  function attach() {{
    doc.querySelectorAll('[data-testid="stSidebar"] [data-testid="stRadio"] label').forEach(l => {{
      const name = l.innerText.trim();
      if (!(name in win.__dsCards) || l.dataset.dsTip) return;
      l.dataset.dsTip = '1';
      l.addEventListener('mouseenter', e => {{ tip.innerHTML = win.__dsCards[name]; tip.style.display = 'block'; move(e); }});
      l.addEventListener('mousemove', move);
      l.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
    }});
  }}
  attach();
  if (!win.__dsTipObs) {{ win.__dsTipObs = new win.MutationObserver(attach);
    win.__dsTipObs.observe(doc.body, {{childList: true, subtree: true}}); }}
}})();</script>""", height=0)



DASH_MARK = ".dashboard_run"
PROTECTED = {"v14m-NIDD-FINAL", "v14m-ISOT-FINAL", "ISOT72-plain-FINAL", "NIDD-plain-FINAL", "v14m-NIDD-CPU-FINAL"}


def stop_run(run_dir, meta, wait=8.0):
    """Stop a run and all its worker processes (own process group), wait until they are gone, mark it stopped."""
    pid = meta.get("pid")
    try:
        os.killpg(pid, signal.SIGTERM)
    except (OSError, TypeError):
        pass
    t0 = time.time()
    while pid and pid_alive(pid) and time.time() - t0 < wait:
        time.sleep(0.2)
    if pid and pid_alive(pid):  # still there: force it
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.5)
    m = load_json(os.path.join(run_dir, "meta.json"), {})
    if m.get("status") in ("preparing", "running"):
        m.update(status="stopped", finished=time.time())
        tmp = os.path.join(run_dir, "meta.json.tmp")
        with open(tmp, "w") as f:
            json.dump(m, f, indent=2)
        os.replace(tmp, os.path.join(run_dir, "meta.json"))


def deletable(run):
    """Only runs started from this dashboard (marker file), never the finals, and only folders inside runs/."""
    d = os.path.realpath(os.path.join(RUNS, run))
    return (run not in PROTECTED and os.path.dirname(d) == os.path.realpath(RUNS)
            and os.path.isfile(os.path.join(d, DASH_MARK)))


def delete_run(run):
    """Delete this run's own folder only (shared caches in data_cache stay: other runs reuse them)."""
    import shutil
    if not deletable(run):
        return False
    shutil.rmtree(os.path.join(RUNS, run), ignore_errors=True)
    if st.session_state.get("run") == run:
        st.session_state["run"] = None
    return not os.path.exists(os.path.join(RUNS, run))


def new_experiment_form():
    ds, cfg_file = chosen_cfg()
    base = load_json(cfg_file)
    d, c, t, j = base["data"], base["clients"], base["training"], base["jiang"]
    rec = json.loads(json.dumps(base))  # the recommended settings, for the "changed" check
    pre = f"nx_f_{ds}_"  # per-dataset widget keys, so one dataset's edits never carry over to another
    with st.sidebar.expander("New experiment", expanded=not list_runs()):
        st.caption(f"Dataset: **{ds}** (switch it below). Train a new run from scratch. Settings start at the ones "
                   "this dataset's final run used and are **locked**; unlock to change them on purpose. "
                   "Results appear under Run below while it trains.")
        edit = st.toggle("Edit settings (advanced)", False, key=f"nx_edit_{ds}",
                         help="Off = the recommended settings (the final run's), shown read-only. Turning it off again, "
                              "pressing Reset, or restarting the app returns every field to them.")
        if not edit or st.button("Reset to recommended", key=f"nx_reset_{ds}", width="stretch"):
            for k in [k for k in st.session_state if str(k).startswith(pre)]:
                del st.session_state[k]
        lock = not edit
        with st.form("new_run"):
            name = st.text_input("Run name", time.strftime("%Y%m%d-%H%M%S"))
            methods = st.multiselect("Methods", METHOD_ORDER, base["methods"], format_func=METHOD_LABELS.get, key=pre + "methods", disabled=lock)
            seeds = st.text_input("Seeds (comma separated)", ", ".join(map(str, base["seeds"])), key=pre + "seeds", disabled=lock)
            st.markdown("**Swarm**")
            num_clients = st.number_input("Drones", 4, 20, c["num_clients"], key=pre + "num_clients", disabled=lock)
            num_rare = st.number_input(app_roles.md("Rare drones", "rare") + f" (own all {d['rare_class']} rows)", 0, 3, c["num_rare"], key=pre + "num_rare", disabled=lock)
            num_poisoned = st.number_input(app_roles.md("Poisoned drones", "poisoned") + " (paper: fewer than half)", 0, 9, c["num_poisoned"], key=pre + "num_poisoned", disabled=lock)
            alpha = st.number_input("Dirichlet α for common data (lower = more skewed)", 0.05, 1000.0,
                                    float(c["dirichlet_alpha"]), key=pre + "alpha", disabled=lock)
            floor = st.slider("Class floor (share of each class spread evenly; paper: every drone has every "
                              "common class)", 0.0, 1.0, float(c.get("class_floor", 0.0)), 0.05, key=pre + "floor", disabled=lock)
            flips = ["attack_to_normal", "rotate"]
            flip = st.selectbox(app_roles.md("Poison", "poisoned") + ": label flip", flips, index=flips.index(c["flip_mode"]), key=pre + "flip", disabled=lock)
            st.markdown("**Training**")
            rounds = st.number_input("FL rounds", 10, 2000, t["rounds"], key=pre + "rounds", disabled=lock)
            steps = st.number_input("Local SGD steps / round", 1, 500, t["local_steps"], key=pre + "steps", disabled=lock)
            opts = ["sgd", "adam"]
            opt = st.selectbox("Local optimiser", opts, index=opts.index(t["optimizer"]), key=pre + "opt", disabled=lock)
            lr = st.number_input("Learning rate", 1e-5, 1.0, float(t["lr"]), format="%.4f", key=pre + "lr", disabled=lock)
            losses = ["restricted_softmax", "ce"]
            loss = st.selectbox("Local loss (honest drones)", losses, index=losses.index(t["local_loss"]), key=pre + "loss", disabled=lock)
            weightings = ["inverse", "sqrt_inverse", "none"]
            cw = st.selectbox("Local class weighting", weightings, index=weightings.index(t["class_weighting"]), key=pre + "cw", disabled=lock)
            st.markdown("**Jiang framework**")
            K = st.number_input("Filter keeps K models", 1, 20, j["K"], key=pre + "K", disabled=lock)
            vstart = st.slider("Voting starts at (fraction of rounds)", 0.0, 0.9, float(j["vote_start_frac"]), 0.05, key=pre + "vstart", disabled=lock)
            margin = st.number_input("Vote margin (standard errors; 0 = exact ≥)", 0.0, 5.0,
                                     float(j.get("vote_margin_se", 0.0)), 0.5, key=pre + "margin", disabled=lock)
            mixes = ["vs_default", "marginal", "aggregation", "eq28"]
            mix = st.selectbox("DQN reward (vs_default = gain over the fixed weight)", mixes,
                               index=mixes.index(j["dqn"]["reward_mix"]) if j["dqn"]["reward_mix"] in mixes else 0, key=pre + "mix", disabled=lock)
            rmetrics = ["macro_f1_prior", "macro_f1", "balanced_accuracy"]
            rmetric = st.selectbox("DQN reward metric (macro_f1_prior = public exam scored as natural traffic)", rmetrics,
                                   index=rmetrics.index(j["dqn"].get("reward_metric", "balanced_accuracy")), key=pre + "rmetric", disabled=lock)
            st.markdown("**Link**")
            st.caption("Training always runs over real mTLS (gRPC, TLS 1.3, X.509): the server and each drone are separate "
                       "processes, and every update is signed (Ed25519) and encrypted (X25519 + AES-256-GCM) by the drone.")
            tamper = st.slider("Simulate interceptions (probability)", 0.0, 0.05, 0.0, 0.005, format="%.3f",
                               help="A simulated attacker on the real mTLS link: share of update packets altered in transit "
                                    "(security.tamper_prob). The server rejects each one (AES-GCM tag check) and the drone "
                                    "re-sends it at the server's request (NACK); the round carries on and results stay "
                                    "identical. Capped at 5% with up to 8 re-sends per packet.")
            confirm = st.checkbox("I changed settings on purpose", False, key=pre + "confirm", disabled=lock,
                                  help="Needed only when a setting differs from the recommended one.")
            go = st.form_submit_button("Start training", type="primary", width="stretch")
    if not go:
        return
    busy = [r for r in list_runs() if run_state(load_json(os.path.join(RUNS, r, "meta.json"), {})) in ("running", "preparing")]
    if busy:  # one training at a time from the dashboard (CPU/RAM sized for one run; avoids double starts)
        st.sidebar.error(f"Already training: {', '.join(busy)}. Stop it first (Stop run or Stop & delete).")
        return
    try:
        seed_list = [int(s) for s in seeds.replace(" ", "").split(",") if s]
    except ValueError:
        st.sidebar.error("Seeds must be whole numbers.")
        return
    if not methods or not seed_list:
        st.sidebar.error("Pick at least one method and one seed.")
        return
    if num_rare + num_poisoned >= num_clients:
        st.sidebar.error("Need at least one regular drone.")
        return
    if "jiang_dqn_pre" in methods and not os.path.exists(pre_store(base)):
        st.sidebar.error("No pre-trained DQN saved yet: use Pre-train DQN first.")
        return
    c.update(num_clients=int(num_clients), num_rare=int(num_rare), num_poisoned=int(num_poisoned),
             dirichlet_alpha=alpha, class_floor=floor, flip_mode=flip)
    t.update(rounds=int(rounds), local_steps=int(steps), optimizer=opt, lr=lr, local_loss=loss, class_weighting=cw)
    j.update(K=int(min(K, num_clients)), vote_start_frac=vstart, vote_margin_se=margin)
    j["dqn"]["reward_mix"] = mix
    j["dqn"]["reward_metric"] = rmetric
    base.update(methods=ordered(methods), seeds=seed_list)
    changed = [f"{sec}.{k}: {rec[sec][k]} → {base[sec][k]}" for sec in ("clients", "training", "jiang")
               for k in rec[sec] if not isinstance(rec[sec][k], dict) and base[sec].get(k) != rec[sec][k]]
    changed += [f"jiang.dqn.{k}: {rec['jiang']['dqn'][k]} → {base['jiang']['dqn'][k]}" for k in ("reward_mix", "reward_metric")
                if base["jiang"]["dqn"].get(k) != rec["jiang"]["dqn"].get(k)]
    changed += [f"{k}: {rec[k]} → {base[k]}" for k in ("methods", "seeds")
                if base[k] != (ordered(rec[k]) if k == "methods" else rec[k])]
    if changed and not confirm:
        st.sidebar.error("These settings differ from the recommended ones: " + "; ".join(changed) +
                         ". Tick “I changed settings on purpose” to train with them, or press Reset.")
        return
    # always real mTLS (separate drone processes, gRPC over TLS 1.3) with signed + encrypted updates
    base.setdefault("security", {}).update(transport="mtls", secure_link=True)
    if tamper > 0:  # simulated attacker: packets altered in transit, recovered by NACK + re-send
        base["security"].update(tamper_prob=float(tamper), max_retries=8)
    run_dir = os.path.join(RUNS, name.strip().replace("/", "_"))
    if os.path.exists(run_dir):
        st.sidebar.error("A run with that name already exists.")
        return
    os.makedirs(run_dir)
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(base, f, indent=2)
    with open(os.path.join(run_dir, DASH_MARK), "w") as f:  # created from the dashboard: may be deleted from it
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    workers = (base.get("compute") or {}).get("workers")  # jobs at once, sized for this machine's RAM
    if (base.get("security") or {}).get("transport") == "mtls":
        # each mTLS job already runs its 10 drones as parallel processes: one job at a time keeps ~11 processes
        # (not ~110) and leaves the dashboard responsive while it trains
        workers = 1
    proc = spawn(["--config", cfg_path, "--out", run_dir] + (["--workers", str(workers)] if workers else []),
                 os.path.join(run_dir, "log.txt"))
    # status file right away (the run overwrites it once its data is ready), so the new run shows at once with its
    # Stop buttons instead of after a delay
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({"status": "preparing", "started": time.time(), "config": base, "methods": base["methods"],
                   "seeds": base["seeds"], "rounds": base["training"]["rounds"], "pid": proc.pid, "errors": []}, f, indent=2)
    st.session_state["run"] = os.path.basename(run_dir)
    time.sleep(0.5)
    st.rerun()


def pre_store(cfg):
    """The pre-trained DQN file the runner loads for this setup (fldos.runner.dqn_store_path, mode setup)."""
    from fldos.runner import dqn_store_path
    return dqn_store_path(cfg, ROOT, "setup")


def pretrain_box():
    with st.sidebar.expander("Pre-train DQN"):
        ds, cfg_file = chosen_cfg()
        st.caption("Carries one DQN through sequential runs on non-final seeds (never seeds 1-5) and saves it for the "
                   f"*Jiang + pre-trained DQN* method. Dataset: **{ds}**, with the recommended settings (as the finals).")
        seeds = st.text_input("Pre-training seeds", "103, 104, 105, 106, 107, 108, 109, 110", key=f"pre_seeds_{ds}",
                              help="The finals used 103-110. Drones train in parallel (identical results, checked).")
        if st.button("Pre-train", width="stretch"):
            os.makedirs(RUNS, exist_ok=True)
            spawn(["--config", cfg_file, "--pretrain-dqn", "--pretrain-mode", "setup", "--seeds",
                   *seeds.replace(" ", "").split(","), "--set", "compute.parallel_drones=10",
                   "--set", "compute.pretrain_eval_every=10"], os.path.join(RUNS, "dqn_pretrain.log"))
            st.success("Started; progress in runs/dqn_pretrain.log")
        store = pre_store(load_json(cfg_file))
        if os.path.exists(store):
            st.caption(f"Saved DQN for this setup: {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(store)))}")
        else:
            st.caption("No pre-trained DQN for this setup yet.")


# ---------- shared bits ----------
def pick(label, options, key, fmt=None, default=None):
    idx = options.index(default) if default in options else 0
    return st.radio(label, options, index=idx, format_func=fmt or str, horizontal=True, key=key)


TAB_INTRO = {
    "overview": "The headline comparison on the test set: does Jiang keep the rare drone (unlike Multi-Krum) while keeping "
                "poisoned drones out (unlike FedAvg)? Each chart below says what it should show.",
    "monitor": "Watch one method and seed round by round: the global model's scores and what the filter, vote and DQN "
               "decided for each drone.",
    "decisions": "Every drone's decisions over the whole run: how often it was selected, voted back in or voted out, and "
                 "how much say it had in the global model.",
    "curves": "How each method learns over the rounds (validation split, averaged over seeds).",
    "results": "Exact test numbers, significance tests and confusion matrices (where each class's traffic was sent).",
    "data": "How the data was split between drones for the chosen seed, and what was removed during cleaning.",
    "log": "The training log and the exact configuration this run used.",
    "mixed": "A separate, harder test (the results in the other tabs are unchanged). In the standard test every window of "
             "packets comes from one recording, so it holds a single class. Here the test traffic is rebuilt as a drone "
             "would see it: normal traffic with an attack burst mixed in at a random time, and the window statistics are "
             "recomputed over that mix. The trained final models are reused as they are (no retraining).",
}


def note(text):
    """Explanatory grey text: hidden unless 'Show explanations' is on in the sidebar (off by default)."""
    if st.session_state.get("show_notes", False):
        st.caption(app_roles.mark(text))


def intro(key):
    note(TAB_INTRO[key])


def purpose(text):
    note(text[:1].upper() + text[1:])


app_ids.configure(METHOD_LABELS, purpose)
app_server.configure(STATUS, STATUS_HELP, ROLE_NAMES, ROLE_COLORS)


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def grouped_bars(df, value, title, fmt, methods, domain=None):
    """Horizontal bars per method, one bar per drone type, with the value printed (so a 0% bar is still visible)."""
    y = alt.Y("Method:N", sort=[METHOD_LABELS[m] for m in methods], title=None)
    off = alt.YOffset("Drone type:N", sort=list(ROLE_COLORS))
    # Overview keeps its original fixed role colours in both themes (vivid red / green)
    fixed = {k: {"Regular": "#607d8b", "Poisoned": "#c62828"}.get(k, "#2e7d32") for k in ROLE_COLORS}
    x = alt.X(f"{value}:Q", title=title, axis=alt.Axis(format=fmt), scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    base = alt.Chart(df).encode(y=y, yOffset=off)
    bars = base.mark_bar(height=12).encode(
        x=x, color=alt.Color("Drone type:N", scale=alt.Scale(domain=list(ROLE_COLORS), range=list(fixed.values())),
                             legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Method", "Drone type", alt.Tooltip(f"{value}:Q", format=fmt.replace("%", ".1%") if fmt == "%" else fmt)])
    text = base.mark_text(align="left", dx=4, fontSize=11).encode(
        x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=".0%" if fmt == "%" else fmt),
        color=alt.Color("Drone type:N", scale=alt.Scale(domain=list(ROLE_COLORS), range=list(fixed.values())),
                        legend=None))
    return (bars + text).properties(height=alt.Step(14))


# ---------- Overview ----------
def tab_overview(run_dir, meta, methods, seeds):
    intro("overview")
    sums = summaries(run_dir, methods, seeds)
    if not sums:
        st.info("The overview appears as soon as the first method has finished a seed.")
        return
    rare = meta.get("rare_class", "Rare")
    done = [m for m in methods if any((m, s) in sums for s in seeds)]
    n_poison = meta.get("config", {}).get("clients", {}).get("num_poisoned", 0)

    # Verdict cards: the paper's three claims, checked on this run.
    by = lambda m, k: [sums[(m, s)]["test_last10"][k] for s in seeds
                       if (m, s) in sums and k in sums[(m, s)]["test_last10"]]
    jm = next((m for m in ("jiang_dqn", "jiang_dqn_pre", "jiang_fixed") if m in done), None)
    if jm:
        st.markdown(f"#### Does {METHOD_LABELS[jm]} do what Jiang et al. claim? (test set, {len(by(jm, 'macro_f1'))} seeds)")
        cols = st.columns(4)
        role = lambda r, k: [sums[(jm, s)][r][k] for s in seeds
                             if (jm, s) in sums and sums[(jm, s)].get(r) and sums[(jm, s)][r].get(k) is not None]
        p_in = role("poisoned", "readmission_rate")
        r_in = role("rare", "readmission_rate")
        g_in = role("general", "readmission_rate")
        cols[0].metric("Poisoned drones voted back in", f"{np.mean(p_in):.0%}" if p_in else "–",
                       help="Should be 0%: the vote keeps attackers out.")
        cols[1].metric("Rare drone voted back in", f"{np.mean(r_in):.0%}" if r_in else "–",
                       help="Should be high: the vote rescues the drone the filter drops for being different.")
        cols[2].metric("Regular drones voted back in", f"{np.mean(g_in):.0%}" if g_in else "–",
                       help="Should be high: honest drones the filter drops are readmitted.")
        cols[3].metric(f"{rare} F1", f"{np.mean(by(jm, 'rare_f1')):.2f}",
                       delta=f"{np.mean(by(jm, 'rare_f1')) - np.mean(by('multi_krum', 'rare_f1')):+.2f} vs Multi-Krum"
                       if "multi_krum" in done else None)
        note("Voted back in = how often the swarm vote readmitted a drone's model after the distance filter dropped "
                   "it (poisoned should be 0%, rare and regular high). " + f"{rare} F1 = how well the rare DoS type "
                   "is detected; only the rare drone ever trains on it.")

    # Table 2: three test views per method.
    st.markdown("#### Jiang Table 2: all classes · the rare class · resistance to poisoning")
    rows = []
    views = [("macro_f1", "All classes (macro F1)", False), ("rare_f1", f"Rare class ({rare} F1)", False),
             ("missed_attack_rate", "Attacks caught (1 − missed)", True)]
    if any(by(m, "false_alarm_rate") for m in done):  # IDS view: normal traffic not flagged
        views.append(("false_alarm_rate", "Normal kept (1 − false alarms)", True))
    for m in done:
        for k, label, invert in views:
            vals = [1 - v if invert else v for v in by(m, k)]
            if not vals:
                continue
            mu, sd = mean_std(vals)
            rows.append({"Method": METHOD_LABELS[m], "View": label, "Score": mu, "lo": max(0, mu - sd),
                         "hi": min(1, mu + sd), "±": sd})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(x=alt.X("Method:N", sort=[METHOD_LABELS[m] for m in done], title=None,
                                        axis=alt.Axis(labels=False, ticks=False)),
                                color=alt.Color("Method:N", scale=method_scale(done),  # legend in bar order, beside the bars
                                                legend=alt.Legend(orient="right", title="Bars, left to right",
                                                                  labelFontSize=13, titleFontSize=12, symbolType="square",
                                                                  symbolSize=220, labelLimit=320, rowPadding=6)))
    bars = base.mark_bar().encode(y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                                  tooltip=["Method", "View", alt.Tooltip("Score:Q", format=".3f"),
                                           alt.Tooltip("±:Q", format=".3f")])
    err = base.mark_errorbar(color=app_ids.ink()).encode(y="lo:Q", y2="hi:Q")
    text = base.mark_text(dy=-8, fontSize=11).encode(y="hi:Q", text=alt.Text("Score:Q", format=".2f"),
                                                       color=alt.value(app_ids.ink()))
    st.altair_chart((bars + err + text).properties(width=alt.Step(34), height=230)
                    .facet(facet=alt.Facet("View:N", title=None, sort=[v[1] for v in views],
                                           header=alt.Header(labelFontSize=13, labelFontWeight="bold")), columns=2)
                    .resolve_scale(x="independent"))
    purpose(f"the paper's main result. FedAvg lets the {n_poison} poisoned drone(s) in, so it misses attacks and its "
            "Normal class is polluted; Multi-Krum blocks the poison but also throws the rare drone away, so the rare "
            "class is lost; Jiang should be high on every panel (the middle ground). The last panel is the IDS "
            "false-alarm view: share of normal traffic not flagged as an attack. Mean of the last 10 global models, "
            "± std over seeds; significance tests are in the Results tab.")

    c1, c2 = st.columns(2)
    # Fig. 6: aggregation weight per drone type.
    w_rows = []
    for m in done:
        for r in ("general", "rare", "poisoned"):
            vals = [sums[(m, s)][r]["mean_weight_share"] for s in seeds
                    if (m, s) in sums and sums[(m, s)].get(r)]
            if vals:
                w_rows.append({"Method": METHOD_LABELS[m], "Drone type": ROLE_NAMES[r], "Share": float(np.mean(vals))})
    with c1:
        st.markdown("#### Jiang Fig. 6: say in the global model, per drone")
        if w_rows:
            st.altair_chart(grouped_bars(pd.DataFrame(w_rows), "Share",
                                         "Average share of the global model (one drone)", "%", done), width="stretch")
        purpose("Jiang's Fig. 6: the poisoned drone's weight should be zero, and the rare drone's weight should "
                "not be below a regular drone's. FedAvg weights everyone by data size, poison included.")
    # Vote outcomes per drone type.
    with c2:
        st.markdown("#### Voting layer: dropped models voted back in")
        v_rows = []
        for m in [m for m in done if m.startswith("jiang")]:
            for r in ("general", "rare", "poisoned"):
                vals = [sums[(m, s)][r]["readmission_rate"] for s in seeds
                        if (m, s) in sums and sums[(m, s)].get(r) and sums[(m, s)][r].get("readmission_rate") is not None]
                if vals:
                    v_rows.append({"Method": METHOD_LABELS[m], "Drone type": ROLE_NAMES[r],
                                   "Voted back in": float(np.mean(vals))})
        if v_rows:
            jms = [m for m in done if m.startswith("jiang")]
            st.altair_chart(grouped_bars(pd.DataFrame(v_rows), "Voted back in",
                                         "Share of votes that readmitted the dropped model", "%", jms, [0, 1.08]),
                            width="stretch")
        else:
            st.info("No Jiang method in this run.")
        purpose("of all the times the filter dropped a drone's model, how often the swarm vote let it back in. "
                "Poisoned should be 0%; rare and regular drones should be high.")

    attack_strength(meta)
    robustness_and_sanity(meta)


SCENARIOS = {"FINAL": "Main: 3 label-flipping drones", "FINAL-poison1": "1 label-flipping drone",
             "FINAL-poison2": "2 label-flipping drones", "ROBUST-scale": "Boosted updates ×10 (Bhagoji 2019)",
             "ROBUST-signflip": "Sign-flipped updates", "ROBUST-collude": "Attackers collude in the vote",
             "SANITY-nopoison": "Sanity: no poisoned drones", "SANITY-norare": "Sanity: no rare drone",
             "STRESS-weakrare": "Stress (pre-declared): rare drone sees only 10% of the De-auth traffic",
             "SCENARIO-steps20": "Light local training (20 steps instead of 80)"}


def same_version(a, b):
    """True when two runs share the version's settings (data, jiang, training) apart from what a scenario varies
    (the threat model in `clients`, and local_steps for the light-training scenario)."""
    strip = lambda c: {k: ({kk: vv for kk, vv in v.items() if kk != "local_steps"} if k == "training" else v)
                       for k, v in c.items() if k in ("data", "jiang", "training")}
    return strip(a) == strip(b)


def robustness_and_sanity(meta):
    """Same seeds and settings, different threat: does Jiang keep its middle-ground behaviour?"""
    rows = []
    for r, label in SCENARIOS.items():
        m = load_json(os.path.join(RUNS, r, "meta.json"), {})
        res = load_json(os.path.join(RUNS, r, "results.json"))
        if not res or not set(m.get("seeds", [])) <= set(meta.get("seeds", [])):
            continue
        if not same_version(m.get("config", {}), meta.get("config", {})):
            continue  # made with another version's settings: not comparable with this run
        for meth in ("fedavg", "multi_krum", "fedwgca", "fed_grab", "jiang_fixed", "jiang_dqn", "jiang_dqn_pre",
                     "jiang_dqn_prexv"):
            e = res.get(meth)
            if not e or "macro_f1_last10" not in e:
                continue
            g = lambda k, f="{:.3f}": f.format(e[k]["mean"]) if k in e else "–"
            rows.append({"Scenario": label, "Seeds": len(m.get("seeds", [])), "Method": METHOD_LABELS[meth],
                         "Macro F1": g("macro_f1_last10"), "DoS detected": g("dos_detection_rate_last10", "{:.1%}"),
                         f"{SHORT.get(meta.get('rare_class', 'De-authentication'), meta.get('rare_class', 'Rare'))} F1":
                             g("rare_f1_last10", "{:.2f}"), "Attacks missed": g("missed_attack_rate_last10", "{:.1%}"),
                         "False alarms": g("false_alarm_rate_last10", "{:.1%}"),
                         "Poison voted back in": g("poisoned_readmission_rate", "{:.0%}"),
                         "Rare voted back in": g("rare_readmission_rate", "{:.0%}")})
    if len({r["Scenario"] for r in rows}) < 2:
        return
    st.markdown("#### Robustness and sanity checks (final seeds)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * (len(rows) + 1) + 3)
    purpose("the same experiment under other threats (stronger model-poisoning attacks, attackers that collude in "
            "the vote) and two sanity checks: with no attackers Jiang should cost nothing, and with no rare drone it "
            "should behave like an ordinary robust method.")


def _strip(cfg):
    cfg = json.loads(json.dumps(cfg or {}))
    for k in ("seeds", "methods", "security", "compute"):  # secure link / CPU-GPU / workers do not change the setup
        cfg.pop(k, None)
    cfg.get("clients", {}).pop("num_poisoned", None)
    return cfg


def attack_strength(meta):
    """Runs that differ from this one only in the number of poisoned drones."""
    ref = _strip(meta.get("config"))
    pts = []
    for r in list_runs():
        m = load_json(os.path.join(RUNS, r, "meta.json"), {})
        res = load_json(os.path.join(RUNS, r, "results.json"))
        if not res or _strip(m.get("config")) != ref or not set(m.get("seeds", [])) <= set(meta.get("seeds", [])):
            continue
        p = m["config"]["clients"]["num_poisoned"]
        for meth, e in res.items():
            if meth in ("fedavg", "multi_krum", "jiang_dqn") and "macro_f1_last10" in e:
                pts.append({"Poisoned drones": p, "Method": METHOD_LABELS[meth], "m": meth,
                            "Attacks missed": e["missed_attack_rate_last10"]["mean"],
                            "Macro F1": e["macro_f1_last10"]["mean"]})
    df = pd.DataFrame(pts).drop_duplicates(["Poisoned drones", "Method"]) if pts else pd.DataFrame()
    if df.empty or df["Poisoned drones"].nunique() < 2:
        return
    st.markdown("#### Attack strength: more poisoned drones")
    ms = [m for m in HEADLINE if m in set(df["m"])]
    df = df.rename(columns={"Poisoned drones": "poisoned", "Attacks missed": "missed", "Macro F1": "mf1"})
    c1, c2 = st.columns(2)
    for col, k, title, fmt in ((c1, "missed", "Attacks missed", ".0%"), (c2, "mf1", "Macro F1", ".2f")):
        col.altair_chart(alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=3).encode(
            x=alt.X("poisoned:Q", title="Poisoned drones (of 10)", axis=alt.Axis(tickMinStep=1, format="d"),
                    scale=alt.Scale(domain=[df["poisoned"].min() - 0.2, df["poisoned"].max() + 0.2])),
            y=alt.Y(f"{k}:Q", title=title, axis=alt.Axis(format=fmt), scale=alt.Scale(zero=k == "missed")),
            color=alt.Color("Method:N", scale=method_scale(ms), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Method", alt.Tooltip("poisoned:Q", title="Poisoned drones"), alt.Tooltip(f"{k}:Q", format=".3f")])
            .properties(height=260), width="stretch", height=330)
    purpose("the same setup re-run with 1, 2 and 3 label-flipping drones (all other settings and seeds equal). "
            "FedAvg gets worse with every poisoned drone; Jiang stays flat because the poisoned drones never get "
            "back in.")


# ---------- Swarm monitor ----------
def tab_monitor(run_dir, meta, methods, seeds):
    intro("monitor")
    c1, c2 = st.columns([3, 1])
    with c1:
        method = pick("Method", methods, "mon_m", METHOD_LABELS.get,
                      next((m for m in methods if m.startswith("jiang_dqn")), None))
    with c2:
        seed = pick("Seed", seeds, "mon_s")
    rounds = read_rounds(run_dir, method, seed)
    if not rounds:
        st.info("Waiting for the first round…")
        return
    follow = st.toggle("Follow latest round", True, key="mon_follow")
    idx = len(rounds) - 1 if follow or len(rounds) == 1 else \
        st.slider("Round", 1, len(rounds), len(rounds), key="mon_round") - 1
    r = rounds[idx]
    v = r["val"] or next((x["val"] for x in reversed(rounds[:idx + 1]) if x.get("val")), None)
    if v is None:
        st.info("No validation score yet for this round.")
        return
    vote_start = meta.get("vote_start_round", 0)
    phase = "voting active" if r["round"] >= vote_start else f"filter only (voting starts at round {vote_start + 1})"
    st.markdown(f"#### Global model after round {r['round'] + 1} / {meta['rounds']} · {phase}")
    cols = st.columns(5)
    cols[0].metric("Macro F1", f"{v['macro_f1']:.1%}")
    cols[1].metric("Accuracy", f"{v['accuracy']:.1%}")
    cols[2].metric(f"{meta.get('rare_class', 'Rare')} F1", f"{v['rare_f1']:.1%}")
    cols[3].metric(f"{meta.get('rare_class', 'Rare')} recall", f"{v['rare_recall']:.1%}")
    cols[4].metric("Attacks missed (called Normal)", f"{v['missed_attack_rate']:.1%}")
    st.caption(f"These five numbers: **validation split, this round only** ({METHOD_LABELS[method]}, seed {seed}). "
               "One round on its own is noisy; the reported result is the test split averaged over the last 10 rounds:")
    summ = load_json(os.path.join(run_dir, method, f"seed_{seed}", "summary.json"))
    if summ and "test_last10" in summ:
        t = summ["test_last10"]
        tc = st.columns(5)
        tc[0].metric("Test macro F1 (last 10)", f"{t['macro_f1']:.1%}")
        tc[1].metric("Test accuracy (last 10)", f"{t['accuracy']:.1%}")
        tc[2].metric(f"Test {meta.get('rare_class', 'Rare')} F1 (last 10)", f"{t['rare_f1']:.1%}")
        tc[3].metric("Test attacks missed (last 10)", f"{t['missed_attack_rate']:.1%}")
        if "false_alarm_rate" in t:
            tc[4].metric("Test false alarms (last 10)", f"{t['false_alarm_rate']:.1%}")

    # Long-run record per drone up to this round (after voting starts) - one round alone is misleading.
    hist = rounds[vote_start:idx + 1]
    longrun = {}
    for d in r["clients"]:
        ds = [x for rr in hist for x in rr["clients"] if x["cid"] == d["cid"]]
        voted = [x for x in ds if "votes_yes" in x]
        longrun[d["cid"]] = {
            "back": np.mean([x["readmitted"] for x in voted]) if voted else np.nan,
            "share": np.mean([x["weight"] for x in ds]) if ds else np.nan}
    # Influence relative to an honest regular drone: 100% = as much say as a typical regular drone, 0% = none.
    reg = [v["share"] for c, v in longrun.items() if r["clients"][c]["role"] == "general" and not pd.isna(v["share"])]
    reg_share = np.mean(reg) if reg else np.nan
    for v in longrun.values():
        v["say"] = v["share"] / reg_share if reg_share and not pd.isna(reg_share) and reg_share > 0 else np.nan

    st.markdown("#### Each drone: this round, and its record so far")
    rows = []
    for d in r["clients"]:
        lr = longrun[d["cid"]]
        rows.append({"Drone": drone(d["cid"]), "Role (ground truth)": ROLE_ICONS[d["role"]],
                     "This round": STATUS[status_key(d, r["round"], vote_start)][0],
                     "Votes on last round's model": f"{d['votes_yes']}/{d['votes_total']}" if d.get("votes_total") else "–",
                     "DQN weight": "–" if d.get("dqn_weight") is None else f"{d['dqn_weight']:.2f}", "Share now": d["weight"],
                     "Say vs a regular drone (so far)": lr["say"], "Voted back in (so far)": lr["back"],
                     "Average share (so far)": lr["share"]})
    df = pd.DataFrame(rows)
    color_of = {name: col for name, col in STATUS.values()}
    fmt_opt = lambda f: (lambda x: "–" if x is None or pd.isna(x) else f.format(x))
    sty = df.style.map(lambda s: f"background-color:{color_of.get(s, '')};color:white;font-weight:600",
                       subset=["This round"])
    st.dataframe(app_roles.rows(sty, [d["role"] for d in r["clients"]], ["Drone", "Role (ground truth)"])
                 .format({"Share now": "{:.1%}",
                          "Say vs a regular drone (so far)": fmt_opt("{:.0%}"), "Voted back in (so far)": fmt_opt("{:.0%}"),
                          "Average share (so far)": fmt_opt("{:.1%}")}),
                 hide_index=True, width="stretch")
    purpose("one round is a snapshot: the filter always sets aside N − K models, so some honest drones are "
            "outside it in any given round. 'Say vs a regular drone' = how much this drone influences the global model "
            "compared with an average honest regular drone (100% = same say, 0% = none; a method can keep a drone yet "
            "give it little say, e.g. FedWGCA down-weights the poisoned drones instead of removing them). Poisoned "
            "drones should be at or near 0%, the rare drone at least 100% for Jiang; every honest drone should be voted "
            "back in most of the time.")
    info = r.get("info") or {}
    if "epsilon" in info:
        st.caption(f"DQN exploration ε = {info['epsilon']:.3f} · decisions so far = {info.get('dqn_decisions')} · "
                   f"last loss = {info.get('dqn_loss') or 0:.5f}")
    st.caption(STATUS_HELP)
    app_ids.why_panel(meta, method, seed, r, rounds, idx)


# ---------- Decisions ----------
def tab_decisions(run_dir, meta, methods, seeds):
    intro("decisions")
    c1, c2 = st.columns([3, 1])
    with c1:
        method = pick("Method", methods, "dec_m", METHOD_LABELS.get,
                      next((m for m in methods if m.startswith("jiang_dqn")), None))
    with c2:
        seed = pick("Seed", seeds, "dec_s")
    rounds = read_rounds(run_dir, method, seed)
    if not rounds:
        st.info("No rounds yet.")
        return
    vs = meta.get("vote_start_round", 0)
    df = pd.DataFrame([{"round": r["round"] + 1, "drone": f"{drone(d['cid'])} {ROLE_ICONS[d['role']]}".strip(),
                        "role": ROLE_NAMES[d["role"]], "status": STATUS[status_key(d, r["round"], vs)][0],
                        "share": d["weight"]} for r in rounds for d in r["clients"]])
    order = [s[0] for s in STATUS.values()]
    colors = {s[0]: s[1] for s in STATUS.values()}
    present = [s for s in order if s in set(df["status"])]
    scale = alt.Scale(domain=present, range=[colors[s] for s in present])
    drone_col = app_roles.label_color({f"{drone(d['cid'])} {ROLE_ICONS[d['role']]}".strip(): d["role"]
                                       for d in rounds[-1]["clients"]})

    st.markdown("#### What happened to each drone (after voting starts)")
    after = df[df["round"] > vs]
    if not after.empty:
        frac = after.groupby(["drone", "status"]).size().reset_index(name="n")
        frac["Fraction"] = frac["n"] / frac.groupby("drone")["n"].transform("sum")
        st.altair_chart(alt.Chart(frac).mark_bar().encode(
            y=alt.Y("drone:N", sort=list(dict.fromkeys(df["drone"])), title=None, axis=alt.Axis(labelColor=drone_col)),
            x=alt.X("Fraction:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of rounds"),
            color=alt.Color("status:N", scale=scale, sort=present, legend=alt.Legend(orient="bottom", title=None)),
            order=alt.Order("order:Q"),
            tooltip=["drone", "status", alt.Tooltip("Fraction:Q", format=".0%")])
            .transform_calculate(order=f"indexof({json.dumps(present)}, datum.status)")
            .properties(height=alt.Step(26)), width="stretch")
        purpose("per drone, the share of rounds spent in each state. Poisoned drones should be solid red "
                "(voted out every time); regular and rare drones should be green/blue with only thin red slivers.")

    st.markdown("#### Say in the global model over time (10-round average)")
    per = df.groupby(["round", "role"], as_index=False)["share"].mean()
    per["share"] = per.groupby("role")["share"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    rule = alt.Chart(pd.DataFrame({"x": [vs + 1]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
    st.altair_chart((alt.Chart(per).mark_line(strokeWidth=2.5).encode(
        x=alt.X("round:Q", title="FL round"), y=alt.Y("share:Q", title="Share per drone", axis=alt.Axis(format="%")),
        color=alt.Color("role:N", scale=alt.Scale(domain=list(ROLE_COLORS), range=list(ROLE_COLORS.values())),
                        legend=alt.Legend(orient="bottom", title=None,
                                          labelColor=app_roles.label_color({ROLE_NAMES[r]: r for r in ROLE_NAMES}))),
        tooltip=["round", "role", alt.Tooltip("share:Q", format=".1%")]) + rule)
        .properties(height=240), width="stretch")
    purpose("after voting starts (dashed line) the poisoned drones' share should drop to 0 and stay there, "
            "while the rare drone keeps a share at least as large as a regular drone's.")

    with st.expander("Round-by-round heatmap (every decision)"):
        st.altair_chart(alt.Chart(df).mark_rect().encode(
            x=alt.X("round:Q", title="FL round", bin=alt.Bin(maxbins=len(rounds))),
            y=alt.Y("drone:N", title=None, sort=list(dict.fromkeys(df["drone"])), axis=alt.Axis(labelColor=drone_col)),
            color=alt.Color("status:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["drone", "round", "status", alt.Tooltip("share:Q", format=".1%")])
            .properties(height=28 * df["drone"].nunique() + 50), width="stretch")
        st.caption(STATUS_HELP)


# ---------- Curves ----------
def tab_curves(run_dir, meta, methods, seeds):
    intro("curves")
    rare = meta.get("rare_class")
    metric_names = {"macro_f1": "Macro F1 (all classes)", "rare_f1": f"{rare} F1 (rare class)",
                    "rare_recall": f"{rare} recall (rare class)",
                    "missed_attack_rate": "Attacks missed (called Normal)", "accuracy": "Accuracy"}
    c1, c2 = st.columns([2, 3])
    with c1:
        metric = st.radio("Metric", list(metric_names), format_func=metric_names.get, key="cur_m")
    with c2:
        default = [m for m in methods if m in HEADLINE + ["fedavg_clean"]] or methods
        show = st.multiselect("Methods", methods, default, format_func=METHOD_LABELS.get, key="cur_methods")
        smooth = st.slider("Smoothing (rounds)", 1, 30, 10, key="cur_smooth")
    rows = []
    for m in show:
        for s in seeds:
            rr = [r for r in read_rounds(run_dir, m, s) if r.get("val")]
            vals = pd.Series([r["val"][metric] for r in rr], dtype=float)
            if vals.empty:
                continue
            for r, v in zip(rr, vals.rolling(smooth, min_periods=1).mean()):
                rows.append({"method": METHOD_LABELS[m], "seed": s, "round": r["round"] + 1, "value": v})
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No rounds yet.")
        return
    agg = df.groupby(["method", "round"])["value"].agg(["mean", "min", "max"]).reset_index()
    vs = meta.get("vote_start_round", 0) + 1
    enc = dict(x=alt.X("round:Q", title="FL round"),
               color=alt.Color("method:N", scale=method_scale(ordered(show)), legend=alt.Legend(orient="bottom", title=None)))
    band = alt.Chart(agg).mark_area(opacity=0.15).encode(
        x=enc["x"], y="min:Q", y2="max:Q", fill=alt.Fill("method:N", scale=method_scale(ordered(show)), legend=None))
    line = alt.Chart(agg).mark_line(strokeWidth=2.5).encode(
        y=alt.Y("mean:Q", title=metric_names[metric], axis=alt.Axis(format="%" if "rate" in metric else ".2f")),
        tooltip=["method", "round", alt.Tooltip("mean:Q", format=".3f")], **enc)
    rule = alt.Chart(pd.DataFrame({"x": [vs]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
    st.altair_chart((line + band + rule).properties(height=400, title=f"{metric_names[metric]} · validation, "
                                                                        f"mean over {len(seeds)} seeds"),
                    width="stretch")
    purpose("Jiang's Fig. 5. Before voting starts (dashed line) Jiang is filter-only, so it tracks Multi-Krum; "
            "once voting and DQN weighting start it climbs to the no-attacker FedAvg line, while FedAvg, carrying the "
            "poisoned drones, sags and varies widely between seeds. Validation is a different time slice from the test "
            "set, so its level differs from the test numbers in the Results tab. Switch the metric to see the rare class (Multi-Krum "
            f"stays at 0) and attacks missed (FedAvg only). Lines = {smooth}-round moving average; shaded = range across seeds.")
    dqn_gap_panel(run_dir, meta, methods, seeds, metric, metric_names[metric], smooth)


def dqn_gap_panel(run_dir, meta, methods, seeds, metric, metric_name, smooth):
    """DQN weight minus fixed weight, round by round on the same seeds (paired): small leads that are invisible
    on the full curves above, and how many seeds each lead holds on."""
    dq = [m for m in ("jiang_dqn", "jiang_dqn_pre", "jiang_dqn_prexv") if m in methods]
    if "jiang_fixed" not in methods or not dq:
        return
    sign = -1 if "missed" in metric or "false_alarm" in metric else 1  # plotted so that up = DQN better
    rate = "rate" in metric
    fmt = (lambda x: f"{x * 100:+.2f} pp") if rate else (lambda x: f"{x:+.3f}")
    val = lambda m, s: {r["round"]: r["val"].get(metric) for r in read_rounds(run_dir, m, s) if r.get("val")}
    rows = []
    for m in dq:
        for s in seeds:
            a, b = val(m, s), val("jiang_fixed", s)
            common = [r for r in sorted(set(a) & set(b)) if a[r] is not None and b[r] is not None]
            gap = pd.Series([sign * (a[r] - b[r]) for r in common], dtype=float)
            for r, g, raw in zip(common, gap.rolling(smooth, min_periods=1).mean(), gap):
                rows.append({"variant": METHOD_LABELS[m], "seed": s, "round": r + 1, "gap": g, "raw": raw})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    st.markdown(f"##### DQN against the fixed weight: {metric_name}, DQN minus fixed (same seeds)")
    agg = df.groupby(["variant", "round"])["gap"].agg(["mean", "min", "max"]).reset_index()
    vs = meta.get("vote_start_round", 0) + 1
    scale = method_scale(ordered(dq))
    enc = dict(x=alt.X("round:Q", title="FL round"),
               color=alt.Color("variant:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)))
    line = alt.Chart(agg).mark_line(strokeWidth=2.5).encode(
        y=alt.Y("mean:Q", title="DQN better ↑   ·   fixed better ↓", axis=alt.Axis(format="+.1%" if rate else "+.2f")),
        tooltip=["variant", "round", alt.Tooltip("mean:Q", format="+.3f"), alt.Tooltip("min:Q", format="+.3f"),
                 alt.Tooltip("max:Q", format="+.3f")], **enc)
    band = alt.Chart(agg).mark_area(opacity=0.15).encode(
        x=enc["x"], y="min:Q", y2="max:Q", fill=alt.Fill("variant:N", scale=scale, legend=None))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=app_ids.ink()).encode(y="y:Q")
    rule = alt.Chart(pd.DataFrame({"x": [vs]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
    st.altair_chart((band + line + zero + rule).properties(
        height=300, title=f"validation, mean over {len(seeds)} seeds; shaded = range across seeds"), width="stretch")
    what = {"jiang_dqn": "DQN that learns during this run",
            "jiang_dqn_pre": "DQN pre-trained before this run",
            "jiang_dqn_prexv": "DQN pre-trained and carried over from earlier versions"}
    keys = " · ".join(f"<span style='color:{METHOD_COLORS[m]};font-size:1.2em'>━━</span> <b>{METHOD_LABELS[m]}</b>: "
                      f"{what[m]}" for m in dq)
    st.markdown(f"<div style='font-size:0.9em;line-height:1.7'>{keys}<br>Each coloured line = that DQN minus "
                f"<b>Jiang, fixed weight</b> on the same seed · solid line at 0 = no difference · grey dashed line = "
                f"voting starts (round {vs}) · shaded = worst to best seed</div>", unsafe_allow_html=True)
    # per round block: mean gap and in how many seeds the DQN is ahead (block mean of the unsmoothed gap)
    last = int(df["round"].max())
    edges = sorted({vs, vs + 50, *range(vs + 150, last, 150), last + 1})
    tab = []
    for v in df["variant"].unique():
        row = {"Variant": v}
        for lo, hi in zip(edges, edges[1:]):
            blk = df[(df["variant"] == v) & (df["round"] >= lo) & (df["round"] < hi)]
            if blk.empty:
                continue
            per_seed = blk.groupby("seed")["raw"].mean()
            row[f"rounds {lo}-{hi - 1}"] = f"{fmt(per_seed.mean())} ({int((per_seed > 0).sum())}/{len(per_seed)} seeds ahead)"
        tab.append(row)
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
    purpose(f"Each seed's DQN run is compared with the fixed-weight run of the same seed (same data, drones and "
            f"attackers), so only the weighting differs. Above 0 = the DQN is ahead on {metric_name}. Dashed line = "
            "voting starts (before it the DQN has nothing to weigh, so the gap is 0). Jiang's claim is that the fixed "
            "weight needs more rounds to collect the rare drone's knowledge, so the DQN should lead early and the gap "
            "may close later. The table gives the block mean and in how many seeds the DQN was ahead, so a lead carried "
            f"by one seed shows up as such. Line = {smooth}-round moving average.")


# ---------- Results ----------
def tab_results(run_dir, meta, methods, seeds):
    intro("results")
    res = load_json(os.path.join(run_dir, "results.json"))
    rare = meta.get("rare_class", "Rare")
    sums = summaries(run_dir, methods, seeds)
    if res:
        st.markdown("#### Test set, mean ± std over seeds (average of the last 10 global models)")
        pm = lambda e, k: f"{e[k]['mean']:.3f} ± {e[k]['std']:.3f}" if k in e else "–"
        pct = lambda e, k: f"{e[k]['mean']:.0%}" if k in e else "–"

        def gen(m, k):
            vals = [sums[(m, s)]["general"][k] for s in seeds if (m, s) in sums and sums[(m, s)].get("general")
                    and sums[(m, s)]["general"].get(k) is not None]
            return f"{np.mean(vals):.0%}" if vals else "–"
        st.dataframe(app_roles.columns(pd.DataFrame([{
            "Method": METHOD_LABELS[m], "Macro F1": pm(e, "macro_f1_last10"), f"{rare} F1": pm(e, "rare_f1_last10"),
            "Accuracy": pm(e, "accuracy_last10"), "Attacks missed": pm(e, "missed_attack_rate_last10"),
            "False alarms": pm(e, "false_alarm_rate_last10"), "Detection rate": pm(e, "detection_rate_last10"),
            "Poison voted back in": pct(e, "poisoned_readmission_rate"), "Poison share": pct(e, "poisoned_mean_weight_share"),
            "Rare voted back in": pct(e, "rare_readmission_rate"), "Rare share": pct(e, "rare_mean_weight_share"),
            "Regular voted back in": gen(m, "readmission_rate"), "Regular share": gen(m, "mean_weight_share")}
            for m, e in res.items()])), hide_index=True, width="stretch")
        note("'Voted back in' = of the votes held on that drone type's dropped models, the share that "
                   "readmitted them (Jiang only). 'Share' = average weight of one such drone in the global model, "
                   "after voting starts.")
        trend_panels(res, rare)
        dos_tables(run_dir, methods, seeds, sums)
        significance_table(run_dir, methods)
        out_of_time_table(run_dir, methods)
        base_rate_table(methods, seeds, sums)
    else:
        st.info("The summary table appears when every job has finished.")
    if not sums:
        return

    last10 = all("confusion_last10" in sums[k] for k in sums)
    st.markdown(f"#### Confusion matrices: test set, summed over {len(seeds)} seeds"
                + (" and the last 10 global models" if last10 else " (final global model)") + ", % of each true class")
    have = [m for m in methods if any((m, s) in sums for s in seeds)]
    show = st.multiselect("Methods", have, [m for m in HEADLINE if m in have] or have[:3],
                          format_func=METHOD_LABELS.get, key="res_methods")
    cols = st.columns(min(3, len(show)) or 1)
    for k, m in enumerate(show):
        mats = [np.array(sums[(m, s)]["confusion_last10"] if last10 else sums[(m, s)]["test"]["confusion"])
                for s in seeds if (m, s) in sums]
        names = sums[next((m, s) for s in seeds if (m, s) in sums)]["class_names"]
        cm = sum(mats)
        cmdf = pd.DataFrame([{"true": names[i], "pred": names[j], "count": int(cm[i, j]),
                              "pct": cm[i, j] / max(1, cm[i].sum()),
                              "label": f"{cm[i, j] / max(1, cm[i].sum()):.0%}\n{int(cm[i, j]):,}"}
                             for i in range(len(names)) for j in range(len(names))])
        base = alt.Chart(cmdf).encode(x=alt.X("pred:N", sort=names, title="Predicted"),
                                      y=alt.Y("true:N", sort=names, title="True"))
        heat = base.mark_rect().encode(color=alt.Color("pct:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                                                       legend=None),
                                       tooltip=["true", "pred", "count", alt.Tooltip("pct:Q", format=".1%")])
        txt = base.mark_text(fontSize=9, lineBreak="\n").encode(text="label:N",
                                                  color=alt.condition(alt.datum.pct > 0.5, alt.value("white"),
                                                                      alt.value("black")))
        box = alt.Chart(pd.DataFrame({"true": [rare]})).mark_rect(fill=None, stroke="#2e7d32", strokeWidth=3) \
            .encode(y=alt.Y("true:N", sort=names))
        normal = meta.get("config", {}).get("data", {}).get("normal_class", "Normal")
        miss = alt.Chart(pd.DataFrame({"pred": [normal]})).mark_rect(fill=None, stroke="#c62828", strokeWidth=2,
                                                                       strokeDash=[4, 2]) \
            .encode(x=alt.X("pred:N", sort=names))
        cols[k % len(cols)].altair_chart((heat + txt + box + miss).properties(title=METHOD_LABELS[m], height=alt.Step(34),
                                                                         width=alt.Step(40))
                                         .configure_axisX(labelAngle=-45, labelLimit=140, labelOverlap=False))
    purpose(f"red dashed column = traffic predicted Normal: for the attack rows these are missed attacks (where "
            f"poisoning shows up). Each row is a true class; the diagonal is correct detection. Green-boxed {rare} row: Multi-Krum "
            f"never lets the rare drone in, so {rare} traffic is misread as another attack; FedAvg and Jiang keep "
            "it on the diagonal (as in the paper, FedAvg keeps the rare class). 'Normal' column: FedAvg lets the "
            "poisoned drones in, so real attacks are waved through as Normal; Jiang's column stays clean.")


def trend_panels(res, rare):
    """Two compact views of the expected trends: damage from poisoning (missed attacks above the attack-free
    reference) with rare-class detection per method, and the DQN variants side by side."""
    m_of = lambda e, k: e[k]["mean"] if isinstance(e.get(k), dict) else (e[k] if isinstance(e.get(k), (int, float)) else np.nan)
    ref = res.get("fedavg_clean")
    base = m_of(ref, "missed_attack_rate_last10") if ref else np.nan
    order = [m for m in ("multi_krum", "fedwgca", "fedavg", "fed_grab", "jiang_fixed", "jiang_dqn", "jiang_dqn_pre",
                         "jiang_dqn_prexv", "fedavg_clean") if m in res]
    c1, c2 = st.container(), st.container()  # stacked: each table needs the full width
    with c1:
        st.markdown(f"##### Poison damage and {rare} detection")
        st.dataframe(pd.DataFrame([{
            "Method": METHOD_LABELS[m],
            "Missed above attack-free (pp)": (m_of(res[m], "missed_attack_rate_last10") - base) * 100,
            f"{rare} recall": m_of(res[m], "rare_recall_last10"), f"{rare} F1": m_of(res[m], "rare_f1_last10"),
            "Poison share": m_of(res[m], "poisoned_mean_weight_share")} for m in order])
            .style.format({"Missed above attack-free (pp)": "{:+.1f}", f"{rare} recall": "{:.3f}", f"{rare} F1": "{:.3f}",
                           "Poison share": "{:.1%}"}, na_rep="–").pipe(app_roles.columns), hide_index=True, width="stretch")
        note("Missed above attack-free = the method's missed-attack rate minus that of FedAvg trained without "
             "attackers (same data): the damage the poisoned drones do. Robust methods should be near 0; FedAvg and "
             f"Fed-GraB above it. {rare} recall/F1 shows whether the rare drone's knowledge reaches the global model.")
    dq = [m for m in ("jiang_fixed", "jiang_dqn", "jiang_dqn_pre", "jiang_dqn_prexv") if m in res]
    if len(dq) > 1:
        with c2:
            st.markdown("##### DQN variants (Jiang's weighting)")
            st.dataframe(pd.DataFrame([{
                "Variant": METHOD_LABELS[m], f"{rare} recall": m_of(res[m], "rare_recall_last10"),
                f"{rare} F1": m_of(res[m], "rare_f1_last10"), "Macro F1": m_of(res[m], "macro_f1_last10"),
                "Missed": m_of(res[m], "missed_attack_rate_last10"), "Rare share": m_of(res[m], "rare_mean_weight_share")}
                for m in dq]).style.format({f"{rare} recall": "{:.3f}", f"{rare} F1": "{:.3f}", "Macro F1": "{:.3f}",
                                            "Missed": "{:.2%}", "Rare share": "{:.1%}"}, na_rep="–")
                .pipe(app_roles.columns), hide_index=True, width="stretch")
            note("Fixed weight → online DQN → pre-trained DQN → DQN carried across versions: more experience should "
                 "give the rare drone more say and raise rare-class detection while poison stays out.")


def out_of_time_table(run_dir, methods):
    """Final models on traffic after the test window (never trained or tested on) vs the test window itself."""
    o = load_json(os.path.join(run_dir, "out_of_time.json"), {}) or {}
    used = {s: d for s, d in o.items() if any(v["future"]["_all_attacks"]["rows"] for v in d.values())}
    if not used:
        return
    st.markdown("#### Out-of-time test: traffic after the test window")
    rows = []
    for m in methods:
        pairs = [(d[m]["test"], d[m]["future"]) for d in used.values() if m in d]
        if not pairs:
            continue
        t = np.mean([a["_all_attacks"]["detected_as_attack"] for a, _ in pairs])
        f = np.mean([b["_all_attacks"]["detected_as_attack"] for _, b in pairs])
        row = {"Method": METHOD_LABELS[m], "Attacks detected · test window": t, "Attacks detected · out-of-time": f,
               "Change": f - t}
        for c in [k for k in pairs[0][1] if not k.startswith("_")]:
            tv = [a[c]["recall"] for a, b in pairs if c in a and c in b and b[c]["rows"]]
            fv = [b[c]["recall"] for a, b in pairs if c in a and c in b and b[c]["rows"]]
            if fv:
                row[f"{c} recall"] = f"{np.mean(tv):.2f} → {np.mean(fv):.2f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).style.format({"Attacks detected · test window": "{:.1%}",
                                                  "Attacks detected · out-of-time": "{:.1%}", "Change": "{:+.1%}"}),
                 hide_index=True, width="stretch")
    n_rows = sum(next(iter(d.values()))["future"]["_all_attacks"]["rows"] for d in used.values())
    st.caption(f"Seeds with later traffic: {', '.join(sorted(used))} ({n_rows:,} attack rows in total; a seed whose test "
               "window already reaches the end of the recording has none). The final global model of each seed is scored "
               "on its test window and on the later slice, both as per-capture windows (the later slice has no Normal, so "
               "no mixed stream or false alarms). A small change means detection holds on newer traffic (no drift).")


def base_rate_table(methods, seeds, sums):
    """Alert precision when attacks are rare (base-rate fallacy, Axelsson 2000): from each method's measured detection
    and false-alarm rates on the mixed test stream (last 10 models, mean over seeds)."""
    rows = []
    for m in methods:
        tl = [sums[(m, s)]["test_last10"] for s in seeds if (m, s) in sums and sums[(m, s)].get("test_last10")]
        cms = [np.array(sums[(m, s)]["confusion_last10"]) for s in seeds if (m, s) in sums and "confusion_last10" in sums[(m, s)]]
        if not tl or not cms:
            continue
        names = sums[next((m, s) for s in seeds if (m, s) in sums)]["class_names"]
        nt = names.index("Normal") if "Normal" in names else 0
        cm = sum(cms)
        prev = 1 - cm[nt].sum() / cm.sum()  # attack share of the test stream
        tpr = float(np.mean([t["detection_rate"] for t in tl]))
        fpr = float(np.mean([t["false_alarm_rate"] for t in tl]))
        meas = float(np.mean([t["binary_precision"] for t in tl])) if all("binary_precision" in t for t in tl) else np.nan
        prec = lambda pi: tpr * pi / (tpr * pi + fpr * (1 - pi)) if tpr * pi + fpr * (1 - pi) > 0 else np.nan
        f1 = lambda pi: 2 * prec(pi) * tpr / (prec(pi) + tpr) if prec(pi) + tpr > 0 else np.nan
        fam = [t.get("false_alarms_per_min") for t in tl if t.get("false_alarms_per_min") is not None]
        rows.append({"Method": METHOD_LABELS[m], "Detection rate": tpr, "False-alarm rate": fpr,
                     "Attack share (test stream)": prev, "Precision measured": meas, "Precision formula @ test share": prec(prev),
                     "Precision @ 10% attacks": prec(0.10), "Precision @ 1%": prec(0.01), "Precision @ 0.1%": prec(0.001),
                     "F1 @ 1%": f1(0.01), "False alarms / hour": 60 * float(np.mean(fam)) if fam else np.nan})
    if not rows:
        return
    st.markdown("#### Realistic base rate: alert precision when attacks are rare")
    df = pd.DataFrame(rows)
    pct = {k: "{:.1%}" for k in df.columns if k not in ("Method", "False alarms / hour")}
    st.dataframe(df.style.format({**pct, "False alarms / hour": "{:,.0f}"}, na_rep="–"), hide_index=True, width="stretch")
    gap = (df["Precision measured"] - df["Precision formula @ test share"]).abs().max()
    st.caption("Precision at attack share π = TPR·π / (TPR·π + FPR·(1−π)), with each method's measured detection rate (TPR) "
               "and false-alarm rate (FPR); detection rate and false alarms per hour do not depend on how rare attacks "
               f"are. Check: at the test stream's own attack share the formula reproduces the measured precision "
               f"(largest difference {gap:.1%}). As expected from the base-rate fallacy (Axelsson 2000; Arp et al. 2022), "
               "precision falls steeply as attacks become rare, fastest for methods with more false alarms.")


def dos_tables(run_dir, methods, seeds, sums):
    """Main-goal view (DoS) and, for mixed-traffic runs, IDS event metrics and the clean per-capture reference."""
    get = lambda m, part, k: [sums[(m, s)][part][k] for s in seeds if (m, s) in sums and k in sums[(m, s)].get(part, {})]
    rows = []
    for m in methods:
        v = lambda k, part="test_last10": get(m, part, k)
        if not v("dos_detection_rate"):
            continue
        f = lambda k, fmt, part="test_last10": (fmt.format(np.mean(v(k, part))) if v(k, part) else "–")
        rows.append({"Method": METHOD_LABELS[m], "DoS detected": f("dos_detection_rate", "{:.1%}"),
                     "DoS type (macro F1)": f("dos_type_macro_f1", "{:.3f}"),
                     "False alarms": f("false_alarm_rate", "{:.1%}"),
                     "Bursts detected": f("event_detection_rate", "{:.0%}"),
                     "Packets to first alert (median)": f("event_delay_packets", "{:.0f}"),
                     "False alarms / min": f("false_alarms_per_min", "{:.1f}"),
                     "False alarms (confirmed alerts)": f("confirmed_false_alarm_rate", "{:.1%}"),
                     "DoS detected (confirmed alerts)": f("confirmed_detection_rate", "{:.1%}"),
                     "False alarms / min (confirmed)": f("confirmed_false_alarms_per_min", "{:.1f}")})
    if rows:
        st.markdown("#### Main goal: DoS detection (test, last 10 global models)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        dos_list = ", ".join(load_json(os.path.join(run_dir, "meta.json"), {}).get("config", {}).get("data", {}).get(
            "dos_classes") or ["DDoS", "UDP", "ICMP", "DoS", "De-authentication"])
        note(f"DoS detected = share of DoS-family traffic ({dos_list}) flagged as an "
                   "attack. DoS type = how well the exact DoS type is named. Bursts detected / packets to first alert / "
                   "false alarms per minute = the IDS view of the mixed-traffic test: each attack burst inserted into "
                   "normal traffic counts once. Confirmed alerts = an alert is raised only when at least 3 of the last 5 "
                   "packets are flagged (standard IDS alert confirmation; removes one-off false alarms).")
    ref = []
    for m in methods:
        v = get(m, "test_pure_last10", "macro_f1")
        if v:
            g = lambda k, fmt: fmt.format(np.mean(get(m, "test_pure_last10", k)))
            ref.append({"Method": METHOD_LABELS[m], "Macro F1": g("macro_f1", "{:.3f}"),
                        "DoS detected": g("dos_detection_rate", "{:.1%}"), "DoS type": g("dos_type_macro_f1", "{:.3f}"),
                        "False alarms": g("false_alarm_rate", "{:.1%}")})
    if ref:
        with st.expander("Clean reference: the same models on the per-capture (unmixed) test"):
            st.dataframe(pd.DataFrame(ref), hide_index=True, width="stretch")
            note("Each attack type on its own, as in the earlier versions of this experiment. Easier than mixed "
                       "traffic, shown only for comparison.")


def significance_table(run_dir, methods):
    """Paired t-test over seeds: is Jiang + DQN's difference from each other method more than seed noise?"""
    if "jiang_dqn" not in methods:
        return
    try:
        sys.path.insert(0, ROOT)
        from significance import compare
    except ImportError:
        return
    run = os.path.basename(run_dir)
    rows = []
    for other in [m for m in methods if m != "jiang_dqn"]:
        for metric, ma, mb, d, p, wins, n, better in compare(run, "jiang_dqn", other):
            rows.append({"Jiang + DQN vs": METHOD_LABELS[other], "Metric": metric, "Jiang + DQN": f"{ma:.4f}",
                         "Other": f"{mb:.4f}", "Difference": f"{d:+.4f}", "p (paired t-test)": f"{p:.4f}",
                         "Significant (p<0.05)": "yes" if p < 0.05 else "no",
                         "Jiang better on": f"{wins}/{n} seeds"})
    if rows:
        st.markdown("#### Significance: Jiang + DQN against each method (paired over seeds)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        note("Same seeds = same data, drones and attackers for every method, so the comparison is paired. "
                   "For missed attacks and false alarms, lower is better.")


# ---------- Mixed-traffic test ----------
def tab_mixed(run_dir, meta, methods):
    intro("mixed")
    res = load_json(os.path.join(run_dir, "mixed_traffic.json"))
    if not res:
        if meta.get("config", {}).get("data", {}).get("traffic") == "mixed":
            st.info("This run was trained and tested on mixed traffic throughout, so its main results (Overview, "
                    "Results) are already the mixed-traffic results. This tab is for older runs trained on pure windows.")
        else:
            st.info("Not computed for this run. From the project folder run:  .venv/bin/python mixed_traffic.py "
                    + os.path.basename(run_dir))
        return
    ep = res["episodes"]
    st.caption(f"{ep['rows']:,} test packets: all Normal test traffic as background, with one attack burst of up to "
               f"{ep['burst_packets']:,} packets per episode, {ep['placements_per_attack']} random placements per attack "
               "type. Final global model of each seed (round 300).")
    rare = meta.get("rare_class", "Rare")
    pm = lambda e, k, f="{:.3f}": f"{f.format(e[k]['mean'])} ± {f.format(e[k]['std'])}"
    have = [m for m in methods if m in res["methods"]]
    st.markdown("#### Mixed traffic: mean ± std over seeds")
    st.dataframe(pd.DataFrame([{"Method": METHOD_LABELS[m], "Macro F1": pm(res["methods"][m], "macro_f1"),
                                "Accuracy": pm(res["methods"][m], "accuracy"),
                                f"{rare} F1": pm(res["methods"][m], "rare_f1", "{:.2f}"),
                                "Attacks missed": pm(res["methods"][m], "missed_attack_rate", "{:.1%}"),
                                "False alarms": pm(res["methods"][m], "false_alarm_rate", "{:.1%}")} for m in have]),
                 hide_index=True, width="stretch")
    purpose("how each method holds up when attack and normal packets share the same windows. Scores are much lower "
            "than in the standard test for every method, because normal packets right after a burst still have attack "
            "packets in their window (false alarms) and the first packets of a burst still look normal (misses). "
            "The comparison should keep its shape: Jiang near the no-attacker reference, FedAvg missing far more "
            "attacks, Multi-Krum still blind to the rare class.")

    st.markdown("#### Confusion matrices: packet counts and % of each true class (summed over seeds)")
    show = st.multiselect("Methods", have, [m for m in HEADLINE if m in have] or have[:3],
                          format_func=METHOD_LABELS.get, key="mixed_methods")
    names = res["class_names"]
    cols = st.columns(min(3, len(show)) or 1)
    for k, m in enumerate(show):
        cm = np.array(res["methods"][m]["confusion"])
        cmdf = pd.DataFrame([{"true": names[i], "pred": names[j], "count": int(cm[i, j]),
                              "pct": cm[i, j] / max(1, cm[i].sum()),
                              "label": f"{cm[i, j] / max(1, cm[i].sum()):.0%}\n{int(cm[i, j]):,}"}
                             for i in range(len(names)) for j in range(len(names))])
        base = alt.Chart(cmdf).encode(x=alt.X("pred:N", sort=names, title="Predicted"),
                                      y=alt.Y("true:N", sort=names, title="True"))
        heat = base.mark_rect().encode(color=alt.Color("pct:Q", scale=alt.Scale(scheme="oranges", domain=[0, 1]),
                                                       legend=None),
                                       tooltip=["true", "pred", "count", alt.Tooltip("pct:Q", format=".1%")])
        txt = base.mark_text(fontSize=9, lineBreak="\n").encode(
            text="label:N", color=alt.condition(alt.datum.pct > 0.5, alt.value("white"), alt.value("black")))
        cols[k % len(cols)].altair_chart((heat + txt).properties(title=METHOD_LABELS[m], height=alt.Step(34),
                                                                 width=alt.Step(40))
                                         .configure_axisX(labelAngle=-45, labelLimit=140, labelOverlap=False))
    purpose("each cell shows the share of that true class and the packet count. Errors cluster where they would in a "
            "real network: normal packets flagged during and just after bursts, and attack packets missed at the start "
            "of a burst.")


# ---------- Data / Log ----------
def tab_data(meta, seeds):
    intro("data")
    parts = meta.get("partitions")
    if not parts:
        st.info("The partition appears once the data is prepared.")
        return
    seed = pick("Seed", seeds, "data_s")
    p = parts.get(str(seed))
    if not p:
        return
    proc = meta.get("config", {}).get("data", {}).get("processed_dir", "data/processed")
    st.caption(f"Saved in `{proc}/seed_{seed}/` (one train + test file per drone, plus validation, test and the "
               f"public exam). Validation {p['n_val']} · test {p['n_test']} · public exam {p['n_public']} rows. "
               "Counts are true labels before any label flipping.")
    clients = p["clients"]
    st.dataframe(app_roles.rows(pd.DataFrame([{"Drone": drone(c["cid"]), "Role": c["role"], "Train rows": c["n_train"],
                                               "Voting set rows": c["n_vote_set"], **c["counts"]} for c in clients]),
                                [c["role"] for c in clients], ["Drone", "Role"]),
                 hide_index=True, width="stretch")
    long = pd.DataFrame([{"Drone": f"{drone(c['cid'])} {ROLE_ICONS[c['role']]}".strip(), "Class": k, "Rows": v}
                         for c in clients for k, v in c["counts"].items()])
    st.altair_chart(alt.Chart(long).mark_bar()
                    .encode(y=alt.Y("Drone:N", sort=None, title=None, axis=alt.Axis(labelColor=app_roles.label_color(
                                {f"{drone(c['cid'])} {ROLE_ICONS[c['role']]}".strip(): c["role"] for c in clients}))),
                            x=alt.X("Rows:Q", stack="normalize", title="Class mix", axis=alt.Axis(format="%")),
                            color=alt.Color("Class:N", legend=alt.Legend(orient="bottom", title=None)),
                            tooltip=["Drone", "Class", "Rows"])
                    .properties(height=30 * len(clients) + 50), width="stretch")
    rare_c = meta.get("rare_class", "De-authentication")
    purpose("Jiang Sec. 5.1: every regular drone holds every common class in different amounts; only the rare "
            f"drone holds {rare_c}. " + ("Normal is only ~2% of UAV-NIDD, so it is a thin slice everywhere."
                                         if dataset_name(meta) == "UAV-NIDD" else
                                         "ISOT has 55 normal recordings, so Normal is a large share everywhere."))
    report = load_json(os.path.join(ROOT, proc, "clean_report.json"))
    if report:
        with st.expander("Cleaning report (dropped columns, duplicates)"):
            st.json(report)
    exact_partition(meta, seed)


def exact_partition(meta, seed):
    """Exact use of the data for this seed (tools/eval_extras.py rebuilds the split with the training code and checks
    it against the drone counts recorded at training time)."""
    rep = (load_json(os.path.join(RUNS, st.session_state.get("run") or "", "partition_report.json"), {}) or {}).get(str(seed))
    st.markdown(f"#### Exact data partition · seed {seed}" + (f" (time fold {rep['fold']})" if rep and rep.get("fold") is not None else ""))
    if not rep:
        st.caption("Not computed for this run yet (`.venv/bin/python tools/eval_extras.py --runs <run>`).")
        return
    classes = rep["classes"]
    tr = {c: sum(d["train"].get(c, 0) for d in rep["drones"]) for c in classes}
    vo = {c: sum(d["vote"].get(c, 0) for d in rep["drones"]) for c in classes}
    total = sum(rep["total"].values())
    p = (meta.get("partitions") or {}).get(str(seed), {})
    rows = [("Drones' local training (80% of each drone's share)", "Local training every round (poisoned drones: labels flipped)", tr),
            ("Drones' voting sets (20% of each drone's share)", "Each drone scores dropped models here when it votes"
             + (" (plus the public exam)" if rep.get("vote_with_public") else ""), vo),
            ("Public exam (server)", "DQN state and reward; feature scaling; added to every voting set", rep["public_exam"]),
            ("Validation", "Round-by-round curves and choosing settings only", rep["validation"]),
            ("Test window", "Final scores (built into the mixed test stream)", rep["test"]),
            ("Out-of-time slice", "Out-of-time test of the final models (Results tab); later than the test window", rep["out_of_time"])]
    df = pd.DataFrame([{"Part": a, "Used for": b, **{c: d.get(c, 0) for c in classes}, "Rows": sum(d.values()),
                        "Share of all rows": sum(d.values()) / total if total else 0} for a, b, d in rows] +
                      [{"Part": "All rows", "Used for": "", **rep["total"], "Rows": total, "Share of all rows": 1.0}])
    st.dataframe(df.style.format({"Share of all rows": "{:.1%}", "Rows": "{:,}", **{c: "{:,}" for c in classes}}),
                 hide_index=True, width="stretch")
    bar = pd.DataFrame([{"Part": a.split(" (")[0], "Rows": sum(d.values()), "order": i} for i, (a, _, d) in enumerate(rows)])
    st.altair_chart(alt.Chart(bar).mark_bar().encode(
        x=alt.X("sum(Rows):Q", stack="normalize", title="Share of all rows", axis=alt.Axis(format="%")),
        color=alt.Color("Part:N", sort=list(bar["Part"]), legend=alt.Legend(orient="bottom", title=None)),
        order=alt.Order("order:Q"), tooltip=["Part", alt.Tooltip("Rows:Q", format=",")]).properties(height=60), width="stretch")
    mixed = p.get("n_test")
    note(f"Attack traffic is split in time (rolling origin): training first, then validation, then the 20% test window; "
         f"traffic after the test window is the out-of-time slice. Normal is split by whole recordings, so it has no "
         f"out-of-time rows. The test window's {sum(rep['test'].values()):,} rows become a mixed test stream of "
         f"{mixed:,} packets (attack bursts inside Normal background" + (", Normal replayed" if mixed and mixed > sum(rep['test'].values()) else "")
         + "). Every row is in exactly one part.")


def tab_log(run_dir, meta):
    intro("log")
    try:
        with open(os.path.join(run_dir, "log.txt")) as f:
            st.code("".join(f.readlines()[-200:]) or "(empty)", language=None)
    except OSError:
        st.caption("No log file (run started from the command line).")
    for e in meta.get("errors", []):
        st.error(e.get("error", ""))
    with st.expander("Config used"):
        st.json(meta.get("config", {}))


# ---------- main ----------
def render_run(run, panel="server"):
    # the live view re-runs on its own every few seconds: always draw the panel picked now, not the one it started with
    panel = st.session_state.get("panel") or panel
    run_dir = os.path.join(RUNS, run)
    meta = load_json(os.path.join(run_dir, "meta.json"), {})
    if not meta:
        st.info("Starting…")
        return
    set_dataset(meta)
    methods, seeds = ordered(meta.get("methods", [])), meta.get("seeds", [])
    state = run_state(meta)
    if state not in ("running", "preparing") and st.session_state.get("live_run") == run:
        st.session_state["live_run"] = None
        st.rerun(scope="app")
    counts = [len(read_rounds(run_dir, m, s)) for m in methods for s in seeds]
    done = sum(counts)
    total = max(1, meta.get("rounds", 1) * len(methods) * len(seeds))
    # Server panel: its own top bar, drawn after the pages so the alert bell counts what the Alerts page holds now
    bar = st.empty() if panel == "server" else app_ids.status_bar(run, meta, dataset_name(meta), state)

    head = st.columns([3, 1, 1])
    badge = app_ids.state_badge(state)
    head[0].markdown(f"### {run} &nbsp; <small>{badge}</small>", unsafe_allow_html=True)
    can_delete = deletable(run)
    if state in ("running", "preparing"):
        if head[1].button("Stop run", width="stretch",
                          help="Stops the run and its workers; files written so far stay (status: stopped)."):
            stop_run(run_dir, meta)
            st.rerun(scope="app")
        if can_delete:
            with head[2].popover("Stop & delete", width="stretch"):
                st.markdown(f"Stop **{run}** and delete its folder `runs/{run}` (rounds, link log, models). "
                            "Shared data caches and every other run stay untouched.")
                if st.button("Stop and delete this run", type="primary", key=f"stopdel_{run}", width="stretch"):
                    stop_run(run_dir, meta)
                    delete_run(run)
                    st.rerun(scope="app")
        st.progress(done / total, text=f"{done}/{total} rounds across {len(methods)} methods × {len(seeds)} seeds")
    elif can_delete:
        with head[2].popover("Delete run", width="stretch"):
            st.markdown(f"Delete **{run}** (started from the dashboard): its folder `runs/{run}` only. "
                        "Shared data caches and every other run stay untouched.")
            if st.button("Delete this run", type="primary", key=f"del_{run}", width="stretch"):
                delete_run(run)
                st.rerun(scope="app")

    if panel == "server":
        server_panel(run_dir, meta, methods, seeds)
        with bar.container():
            app_server.top_bar(run, meta, dataset_name(meta), state, max(counts, default=0))
    elif panel == "sites":
        sites_panel(run_dir, meta, methods, seeds, state in ("running", "preparing"), max(counts, default=0))
    else:
        secure_link_panel(run_dir, meta)


# ---------- panels ----------
# Three consoles over the same run: what the aggregation server sees, what each drone site sees, and the link between.
PANELS = {"server": ":material/dns: Aggregation Server", "sites": ":material/drone: Drone Sites",
          "link": ":material/encrypted: Secure Link"}
PANEL_HELP = ("Aggregation Server: what the server can see (updates, filter, votes, weights, global model) · "
              "Drone Sites: what each site sees (its own traffic, labels and role) · "
              "Secure Link: the encrypted channel between the drones and the server")


def panel_switch():
    return st.segmented_control("Console", list(PANELS), default="server", required=True, format_func=PANELS.get,
                                key="panel", label_visibility="collapsed", help=PANEL_HELP)


def server_panel(run_dir, meta, methods, seeds):
    tabs = st.tabs([":material/dashboard: Overview", ":material/radar: Swarm monitor", ":material/how_to_vote: Decisions",
                    ":material/verified_user: Trust scores", ":material/receipt_long: Audit log",
                    ":material/show_chart: Curves", ":material/table_chart: Results", ":material/notifications: Alerts",
                    ":material/fact_check: Verification", ":material/terminal: Log"])
    with tabs[0]:
        tab_overview(run_dir, meta, methods, seeds)
    with tabs[1]:
        tab_monitor(run_dir, meta, methods, seeds)
    with tabs[2]:
        tab_decisions(run_dir, meta, methods, seeds)
    with tabs[3]:
        app_server.tab_trust(run_dir, meta, methods, seeds)
    with tabs[4]:
        app_server.tab_audit(run_dir, meta, methods, seeds)
    with tabs[5]:
        tab_curves(run_dir, meta, methods, seeds)
    with tabs[6]:
        tab_results(run_dir, meta, methods, seeds)
    with tabs[7]:
        app_ids.tab_alerts(run_dir, meta, methods, seeds)
    with tabs[8]:
        app_ids.tab_verification(run_dir, meta)
    with tabs[9]:
        tab_log(run_dir, meta)


def sites_panel(run_dir, meta, methods, seeds, live, fl_round):
    tabs = st.tabs([":material/map: Site map", ":material/drone: Drone page", ":material/database: Data",
                    ":material/shuffle: Mixed-traffic test", ":material/science: Test", ":material/sensors: Live"])
    with tabs[0]:
        app_sites.tab_site_map(run_dir, meta, live, fl_round)
    with tabs[1]:
        app_sites.tab_drone(run_dir, meta, methods, seeds)
    with tabs[2]:
        tab_data(meta, seeds)
    with tabs[3]:
        tab_mixed(run_dir, meta, methods)
    with tabs[4]:
        app_ids.tab_test(run_dir, meta, methods, seeds)
    with tabs[5]:
        app_ids.tab_live(run_dir, meta, methods, seeds)


def secure_link_panel(run_dir, meta):
    """Hook for the Secure Link visualisation (app_securelink.py, a separate module); a placeholder until it exists."""
    try:
        import app_securelink
    except ImportError:
        app_securelink = None
    # a switch, not tabs: only the selected view is built, so only one animated frame exists at a time
    # the full demo and the encrypted-vs-normal comparison are built from demo_runs/, which uses the UAV-NIDD · CPU-only
    # setup: they are offered under that dataset button only
    views = ["run", "demo", "cost", "certs"] if st.session_state.get("dataset_sel") == DEMO_DATASET else ["run"]
    if "sl_view" in st.session_state and st.session_state["sl_view"] not in views:
        st.session_state["sl_view"] = "run"
    view = st.segmented_control("Secure Link view", views, default=None if "sl_view" in st.session_state else "run",
                                required=True, key="sl_view",
                                format_func={"run": ":material/lan: This run's link",
                                             "demo": ":material/play_circle: Full demo",
                                             "cost": ":material/compare_arrows: Encrypted vs normal",
                                             "certs": ":material/badge: Certificates"}.get,
                                label_visibility="collapsed")
    if views == ["run"]:
        st.caption(f"The full demo, the encrypted-vs-normal comparison and the certificates are under the "
                   f"**{DEMO_DATASET}** dataset button (they use that setup).")
    if app_securelink is None:
        if view == "run":
            secure_link_placeholder(meta)
        else:
            st.info("The step-by-step demo (enrollment, encryption, transmission, decryption, filter, vote, DQN weight, "
                    "broadcast, interception → re-send) appears here once the Secure Link module is installed.")
        return
    try:
        if view == "run":
            app_securelink.render_secure_link(run_dir, meta)
            app_mtls.render_run_panel(run_dir, meta)  # real mTLS runs only (their mtls_meta.json / mtls.jsonl)
        elif view == "demo":
            try:
                tr = demo_trace(MTLS_RUN)
            except Exception as e:  # e.g. a run without the message-layer link log: say so, keep the panel below
                st.warning(f"Could not build the demo from `{os.path.relpath(MTLS_RUN, ROOT)}`: {e}")
                tr = None
            if tr:
                app_securelink.render_demo(tr)
            else:
                st.info("The demo is built from `demo_runs/mtls` (real mTLS run), which is not finished yet.")
            app_mtls.render_demo_panel(os.path.join(MTLS_RUN, "jiang_dqn", "seed_1"))
            demo_sources(tr)
        elif view == "certs":
            app_mtls.render_certificates(app_mtls._roles(load_json(os.path.join(MTLS_RUN, "meta.json"), {})),
                                         run_pkis=[("the Real mTLS demo run (`demo_runs/mtls`)", os.path.join(MTLS_RUN, "pki"))])
        else:
            encryption_compare()
    except Exception as e:  # keep the other consoles usable if the visualisation fails
        st.error(f"Secure Link view failed: {e}")
        st.exception(e)



DEMO_RUNS = os.path.join(ROOT, "demo_runs")  # encryption comparison runs (kept apart from runs/)
DEMO_DATASET = "UAV-NIDD · CPU-only"  # the dataset button whose setup demo_runs/ uses
ENC_RUN = MTLS_RUN = os.path.join(DEMO_RUNS, "mtls")  # the real mTLS run: drone processes seal their own updates
PLAIN_RUN = os.path.join(DEMO_RUNS, "normal")  # same setup, no link, drones in parallel processes too


@st.cache_data(show_spinner="Building the demo from the run's files…", max_entries=4)
def _demo_trace(run_dir, mtime):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import make_demo_trace
    return make_demo_trace.build(run_dir, "jiang_dqn")


def demo_trace(run_dir=ENC_RUN):
    """The full demo's data: read from the real mTLS demo run (demo_runs/mtls) at view time (None until it has finished)."""
    job = os.path.join(run_dir, "jiang_dqn", "seed_1", "summary.json")
    if not os.path.exists(job):
        return None
    return _demo_trace(run_dir, os.path.getmtime(job))


def demo_sources(tr=None):
    """Provenance of every element in the demo: the real run, file and field it is read from, and the raw record.
    The demo is built from the real mTLS run demo_runs/mtls."""
    tr = tr or {}
    m = tr.get("meta", {})
    run = m.get("run_dir", "demo_runs/mtls")
    job = f"{run}/{m.get('method', 'jiang_dqn')}/seed_{m.get('seed', 1)}"
    with st.expander(":material/source: Where this demo's data comes from", expanded=False):
        st.markdown(f"Every number and byte below the Play button comes from **one real training run**: `{run}` "
                    f"({m.get('dataset', 'UAV-NIDD')}, config `{m.get('config_file', '')}`, seed {m.get('seed')}, "
                    f"{app_ids.lab(m.get('method', 'jiang_dqn'))}, {m.get('rounds')} rounds, voting from round "
                    f"{(m.get('vote_start') or 0) + 1}, {m.get('local_steps', 80)} local steps), trained over real mTLS "
                    "(gRPC over TLS 1.3, each drone a separate process with its own X.509 client certificate); the drone "
                    "processes sign and encrypt their own updates (Ed25519, X25519 + AES-256-GCM). Only the attacker "
                    "(interceptions) and the network conditions (localhost, a simulated 10 Mbit/s transfer time) are simulated. "
                    "The same run without the link (`demo_runs/normal`) gives identical rounds and global model (see "
                    "Encrypted vs normal). The demo is built from these files "
                    "each time it is opened (`tools/make_demo_trace.build`); no values are invented.")
        st.dataframe(pd.DataFrame([
            ("1 Enrollment & keys", f"{job}/securelink_meta.json", "enrollment[].fingerprint, sign_pub, kex_pub, ca/server fingerprint, hkdf_salt, enroll_ms", "fldos/securelink.py"),
            ("Sites: roles, class rows", f"{run}/meta.json", "partitions['1'].clients[].role, counts, n_train", "fldos/data.py (partition)"),
            ("Site names / map positions", "app_sites.py", "SITES (display labels only)", "dashboard"),
            ("2 Local training", f"{job}/securelink.jsonl", "uplink[].local_train_ms, update_norm, plain_sample (first 8 weights of the real update)", "fldos/runner.py"),
            ("3 Encryption", f"{job}/securelink.jsonl", "uplink[].plain_hex, nonce_hex, cipher_hex, tag_hex, sig_hex, bytes, enc_ms", "fldos/securelink.py"),
            ("4 Transmission", f"{job}/securelink.jsonl", "uplink[].bytes, transfer_s (simulated 10 Mbit/s, 20 ms transfer time)", "fldos/securelink.py Link"),
            ("5 Decrypt & verify", f"{job}/securelink.jsonl", "uplink[].dec_ms, verify_ok, decrypted_sample (bit-identical check)", "fldos/securelink.py"),
            ("6 Filter", f"{job}/rounds.jsonl", "clients[].status, score (distance), filtered_out", "fldos/aggregators.py Jiang"),
            ("7 Vote", f"{job}/rounds.jsonl + securelink.jsonl", "clients[].yes / votes_total; vote{messages, bytes, ballots}", "fldos/aggregators.py, securelink.py"),
            ("8 DQN weight", f"{job}/rounds.jsonl", "clients[].weight, dqn_weight, reward; info", "fldos/aggregators.py, fldos/dqn.py"),
            ("9 Aggregate & broadcast", f"{job}/securelink.jsonl + rounds.jsonl", "global_sample, downlink[]; val (macro F1, rare recall, missed attacks)", "fldos/runner.py"),
            ("10 Interception → re-send", f"{job}/securelink.jsonl", f"events[] of round {(tr.get('tamper_demo') or {}).get('round', 22) + 1} (simulated attacker, re-sent on request)", "fldos/securelink.py"),
        ] + [
            ("mTLS certificates & handshakes (panel below the demo)", f"{job}/mtls_meta.json", "ca, server, drones[].subject, serial, fingerprint, not_after, tls_version, cipher, handshake_ms, pid, status", "mTLS transport (fldos)"),
            ("mTLS wire traffic", f"{job}/mtls.jsonl", "drones[].bytes_up, bytes_down, rpc_ms_up, rpc_ms_down; server_wait_ms", "mTLS transport (fldos)"),
            ("mTLS run metadata", f"{job}/mtls_meta.json", "see above (certificates, TLS version / cipher, handshake_ms, pid)", "mTLS transport (fldos)"),
        ], columns=["Demo element", "File", "Field(s)", "Produced by"]), hide_index=True, width="stretch")
        c1, c2 = st.columns([1, 2])
        files = {"rounds.jsonl (server decisions)": f"{job}/rounds.jsonl", "securelink.jsonl (link log)": f"{job}/securelink.jsonl"}
        files["mtls.jsonl (wire traffic)"] = f"{job}/mtls.jsonl"
        files["mtls_meta.json (certificates, handshakes)"] = f"{job}/mtls_meta.json"
        which = c1.selectbox("Show the raw record", list(files), key="demo_src_file")
        rnd = c2.slider("Round", 1, int(m.get("rounds", 40)), 1, key="demo_src_round")
        path = os.path.join(ROOT, files[which])
        try:
            with open(path) as f:
                if path.endswith("meta.json"):
                    rec = json.load(f)
                else:
                    rec = next((json.loads(l) for i, l in enumerate(f) if i == rnd - 1), None)
        except OSError:
            rec = None
        if rec is None:
            st.info(f"{files[which]} not found.")
        else:
            st.caption(files[which] if path.endswith("meta.json") else f"{files[which]}, line {rnd} (round {rnd})")
            st.json(rec, expanded=1)


def _cmp(method):
    """Real mTLS run vs the normal run for one method: round times, crypto, wire traffic, identity (files only)."""
    j, jp = os.path.join(MTLS_RUN, method, "seed_1"), os.path.join(PLAIN_RUN, method, "seed_1")
    try:
        rm = [json.loads(l) for l in open(os.path.join(j, "rounds.jsonl"))]
        rp = [json.loads(l) for l in open(os.path.join(jp, "rounds.jsonl"))]
    except OSError:
        return None
    link = []
    if os.path.exists(os.path.join(j, "securelink.jsonl")):
        link = [json.loads(l) for l in open(os.path.join(j, "securelink.jsonl"))]
    wire = app_mtls.read_log(j)
    import torch
    strip = lambda r: {k: v for k, v in r.items() if k != "time_s"}
    try:
        same_model = torch.equal(torch.load(os.path.join(j, "global_model.pt")), torch.load(os.path.join(jp, "global_model.pt")))
    except Exception:
        same_model = None
    return {"rounds": rm, "plain": rp, "link": link, "wire": wire, "meta": load_json(os.path.join(j, app_mtls.META), {}),
            "summary": load_json(os.path.join(j, "summary.json"), {}),
            "same_rounds": [strip(r) for r in rm] == [strip(r) for r in rp], "same_model": same_model}


def encryption_compare():
    """Encrypted vs normal: the same setup and seed trained twice, both with the 10 drones as parallel processes."""
    st.markdown("#### Encrypted (real mTLS) vs normal training (same setup, same seed)")
    meta_e, meta_p = load_json(os.path.join(MTLS_RUN, "meta.json"), {}), load_json(os.path.join(PLAIN_RUN, "meta.json"), {})
    if meta_e.get("status") != "done" or meta_p.get("status") != "done":
        st.info("The comparison runs (`demo_runs/normal`, `demo_runs/mtls`) are not finished yet"
                + (" (the normal run is being created now)." if meta_p.get("status") != "done" else "."))
        return
    st.caption(f"`demo_runs/normal` vs `demo_runs/mtls`: UAV-NIDD recommended setup (class weights, CPU), seed 1, "
               f"{meta_e.get('rounds')} rounds, voting from round {meta_e.get('vote_start_round', 0) + 1}. Both runs "
               "train the 10 drones in parallel processes, so their round times are comparable. The normal run has no link; "
               "the mTLS run uses gRPC over TLS 1.3 with mutual X.509 certificates, and the drone processes sign (Ed25519) and "
               "encrypt (X25519 → HKDF-SHA256 → AES-256-GCM) their own updates.")
    rows, curves = [], []
    for m in ("fedavg", "jiang_dqn"):
        c = _cmp(m)
        if not c:
            continue
        te = np.array([r.get("time_s", 0) for r in c["rounds"]]); tp = np.array([r.get("time_s", 0) for r in c["plain"]])
        if not len(te) or not len(tp):
            continue
        crypto = np.array([(l.get("totals") or {}).get("crypto_ms", 0) for l in c["link"]])
        wire = np.array([sum((d.get("bytes_up") or 0) + (d.get("bytes_down") or 0) for d in x.get("drones") or [])
                         for x in c["wire"]]) / 1e6
        resend = sum(l.get("retransmits", 0) for l in c["link"])
        drones = c["meta"].get("drones") or []
        hs = [d.get("handshake_ms") for d in drones if isinstance(d.get("handshake_ms"), (int, float))]
        ver, cipher = app_mtls.tls_of(c["meta"])
        rows.append({"Method": METHOD_LABELS[m],
                     "Round time, normal (s)": tp.mean(), "Round time, mTLS (s)": te.mean(),
                     "Extra time": te.mean() / tp.mean() - 1 if tp.mean() else np.nan,
                     "Crypto per round (ms)": crypto.mean() if len(crypto) else np.nan,
                     "TLS": " · ".join(x for x in (app_mtls._tls_name(ver) if ver else None, cipher) if x) or "–",
                     "Wire traffic per round (MB)": wire.mean() if len(wire) else np.nan,
                     "Handshake, mean (ms)": float(np.mean(hs)) if hs else np.nan,
                     "Re-sends": resend,
                     "Same rounds": "yes" if c["same_rounds"] else "NO",
                     "Same final model": {True: "yes", False: "NO", None: "–"}[c["same_model"]],
                     "Test macro F1": (c["summary"].get("test_last10") or {}).get("macro_f1")})
        for tag, arr in (("normal", tp), ("mTLS", te)):
            curves += [{"Method": METHOD_LABELS[m], "Run": tag, "round": i + 1, "s": v} for i, v in enumerate(arr)]
    if not rows:
        st.info("No finished jobs in the comparison runs yet.")
        return
    st.dataframe(pd.DataFrame(rows).style.format({
        "Round time, normal (s)": "{:.3f}", "Round time, mTLS (s)": "{:.3f}", "Extra time": "{:+.1%}",
        "Crypto per round (ms)": "{:.1f}", "Wire traffic per round (MB)": "{:.2f}", "Handshake, mean (ms)": "{:.2f}",
        "Test macro F1": "{:.4f}"}, na_rep="–"), hide_index=True, width="stretch")
    cv = pd.DataFrame(curves)
    cv["smooth"] = cv.groupby(["Method", "Run"])["s"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    st.altair_chart(alt.Chart(cv).mark_line(strokeWidth=2).encode(
        x=alt.X("round:Q", title="FL round"), y=alt.Y("smooth:Q", title="Seconds per round (10-round mean)"),
        color=alt.Color("Run:N", scale=alt.Scale(domain=["normal", "mTLS"], range=["#90a4ae", "#1565c0"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.StrokeDash("Method:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Method", "Run", "round", alt.Tooltip("s:Q", format=".3f")]).properties(height=280), width="stretch")
    st.caption("Extra time = mTLS round time / normal round time − 1 (wall clock, same machine, both with the 10 drones "
               "in parallel processes). Crypto = sealing + opening every message of the round (message layer, from "
               "`securelink.jsonl`). Wire traffic = gRPC message bytes from `mtls.jsonl` (TLS records add about 22-29 bytes "
               "each); TLS and handshake from each drone's connection in `mtls_meta.json`. Re-sends = packets altered by the "
               "simulated attacker and sent again on request. Same rounds / final model: every logged round (time "
               "excluded) and the final global model compared with the normal run; identical means the encryption and the "
               "real transport changed nothing in training.")


def secure_link_placeholder(meta):
    sl = meta.get("secure_link") or {}
    st.markdown("#### Encrypted drone ↔ server link")
    cols = st.columns(4)
    for col, (label, k, v) in zip(cols, (("Key exchange", "kex", "X25519"), ("Encryption", "aead", "AES-256-GCM"),
                                         ("Signature", "sig", "Ed25519"), ("Key derivation", "kdf", "HKDF-SHA256"))):
        col.metric(label, sl.get(k, v))
    if sl:
        st.success(f"This run used the secure link ({len(sl.get('enrollment', []))} drones enrolled).")
    else:
        st.caption("This run was trained without the secure link. New runs started from the sidebar use real mTLS with signed, "
                   "encrypted updates (results are identical: each decrypted update is checked bit-identical to the sent one).")
    st.info("The live link view appears here once the Secure Link module is installed.")


def main():
    app_ids.theme_toggle()
    app_ids.inject_css()
    st.sidebar.title("FL DoS Swarm")
    st.sidebar.caption("Jiang et al. (2024) filtering + voting + DQN on UAV-NIDD and ISOT Drone")
    try:
        ver = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%h %cd", "--date=format:%Y-%m-%d %H:%M"],
                             capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        ver = ""
    st.sidebar.caption(f"Code version: {ver or time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(__file__)))}"
                       " · if this looks old, use Restart app server")
    runs = list_runs()
    if not runs:
        new_experiment_form()
        pretrain_box()
        st.title("Federated DoS detection for drone swarms")
        st.info("Start your first experiment from the sidebar.")
        return
    # Dataset switch: the same tabs and charts for either dataset; it filters the runs and opens that dataset's final run.
    # default run per dataset: the v14 finals (v12c / v13 finals stay selectable as FINAL, v12c-FINAL, ISOT-FINAL, ...)
    finals = {"UAV-NIDD": "v14m-NIDD-FINAL", "ISOT Drone": "v14m-ISOT-FINAL", "ISOT Drone · 72 inputs": "ISOT72-plain-FINAL",
              "UAV-NIDD · plain": "NIDD-plain-FINAL", "UAV-NIDD · CPU-only": "v14m-NIDD-CPU-FINAL"}
    ds_of = {r: dataset_name(load_json(os.path.join(RUNS, r, "meta.json"), {})) for r in runs}
    have = [d for d in finals if any(v == d for v in ds_of.values())]
    cur_ds = ds_of.get(st.session_state.get("run"))
    dataset = st.sidebar.radio("Dataset", have, horizontal=True, index=have.index(cur_ds) if cur_ds in have else 0,
                               help="UAV-NIDD: very little normal traffic (0.75%). ISOT Drone: plenty of normal traffic "
                                    "(55 recordings); 29 size/timing inputs with class weights (v14m). ISOT Drone · 72 inputs: "
                                    "the same ISOT data with per-packet protocol fields (TCP flags, protocol, payload), plain "
                                    "loss, all 27,145,265 packets (no FedWGCA / Fed-GraB / pre-trained DQN in this run). UAV-NIDD · plain: "
                                    "the same UAV-NIDD setup without class weights (plain loss); no pre-trained DQN. "
                                    "Fed-GraB never uses class weights (its own balancer), so its runs are the same "
                                    "in both UAV-NIDD entries. UAV-NIDD · CPU-only: v14m with seeds 4-5 re-run on CPU "
                                    "(v14m ran them on the GPU), so every seed uses the same device.") \
        if len(have) > 1 else (have or ["UAV-NIDD"])[0]
    st.session_state["dataset_sel"] = dataset
    dataset_hover(finals, have)
    runs = [r for r in runs if ds_of[r] == dataset] or runs
    current = st.session_state.get("run")
    default = current if current in runs else (finals[dataset] if finals.get(dataset) in runs else runs[0])
    run = st.sidebar.selectbox("Run", runs, index=runs.index(default))
    st.session_state["run"] = run
    st.sidebar.divider()
    st.sidebar.caption("ADMIN")  # training controls kept apart from monitoring, as in IDS consoles
    new_experiment_form()
    pretrain_box()
    st.sidebar.toggle("Show explanations", False, key="show_notes",
                      help="Grey notes under each chart saying what it shows (off = console view).")
    app_roles.toggle()
    if st.sidebar.button("Restart app server (load new code)", width="stretch"):
        # A detached shell starts a fresh server on the same port once this one has exited; the browser tab
        # reconnects by itself. Running experiments are separate processes and keep running.
        port = st.get_option("server.port")
        cmd = (f'sleep 2; cd "{ROOT}"; PYTORCH_ENABLE_MPS_FALLBACK=1 nohup "{ROOT}/.venv/bin/python" -m streamlit run app.py '
               f'--server.port {port} --server.headless true --browser.gatherUsageStats false '
               f'> "{RUNS}/app_server.log" 2>&1 &')
        subprocess.Popen(["/bin/bash", "-c", cmd], start_new_session=True)
        st.sidebar.info("Restarting... the page reconnects in a few seconds (refresh if it doesn't).")
        os.kill(os.getpid(), signal.SIGTERM)
    if st.sidebar.button("Quit app server", width="stretch"):
        st.sidebar.success("Server stopped. You can close this tab.")
        os.kill(os.getpid(), signal.SIGTERM)
    meta = load_json(os.path.join(RUNS, run, "meta.json"), {})
    live = run_state(meta) in ("running", "preparing")
    st.session_state["live_run"] = run if live else None
    panel = panel_switch()
    st.fragment(run_every=3 if live else None)(render_run)(run, panel)


main()
