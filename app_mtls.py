"""Real mTLS link views of the dashboard (app.py; notes/SPEC_P18_MTLS.md, "UI (stream B)").

Read-only, except the Certificates view's Create / Revoke / Restore, which go through the transport's own functions
(fldos/mtls.py ensure_project_pki, set_revoked); this module never writes revoked.json itself.
Everything shown is read from the files the mTLS transport writes:
  <job>/mtls_meta.json   transport, port, CA / server certificate, per-drone connection (subject, serial, fingerprint,
                         not_after, pid, TLS version, cipher, handshake ms, connected_at, status), grpc / OpenSSL versions
  <job>/mtls.jsonl       one line per round: per-drone gRPC message bytes up / down, RPC ms, server wait ms
  pki/                   the project PKI (CA + server + drone certificates, PEM) and revoked.json

  link_label(run_dir, meta)        "mTLS · TLS 1.3 · <cipher>" for the status bars (None when not an mTLS run)
  render_run_panel(run_dir, meta)  Secure Link → This run's link: connection table + wire traffic per round
  render_demo_panel(job_dir)       Full demo (mTLS source): real X.509 fields and TLS handshake data
  render_certificates(roles)       Secure Link → Certificates: project PKI with Revoke / Restore per drone
"""
import datetime as dt
import glob
import json
import os
import re
import time

import altair as alt
import pandas as pd
import streamlit as st

import app_roles

ROOT = os.path.dirname(os.path.abspath(__file__))
PKI_DIR = os.path.join(ROOT, "pki")
META, LOG = "mtls_meta.json", "mtls.jsonl"
REVOKED = "revoked.json"
TLS_DEFAULT = "TLS 1.3"  # the transport's protocol (spec); shown until a job has logged its negotiated version


def drone_name(cid):
    return f"Drone {int(cid) + 1:02d}"


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _stamp(path):
    try:
        s = os.stat(path)
        return s.st_mtime_ns, s.st_size
    except OSError:
        return None


@st.cache_data(show_spinner=False, max_entries=16)
def _jsonl(path, stamp):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # half-written last line while training
                    break
    except OSError:
        pass
    return out


def read_log(job):
    p = os.path.join(job, LOG)
    return _jsonl(p, _stamp(p)) if os.path.exists(p) else []


def transport(meta):
    return (((meta or {}).get("config") or {}).get("security") or {}).get("transport") or "inproc"


def is_mtls(meta):
    return transport(meta) == "mtls"


def jobs(run_dir, meta=None):
    """[(method, seed, job_dir)] of this run that have an mtls_meta.json, in the run's method order; a run-level
    mtls_meta.json counts as one job ("", "")."""
    if not run_dir or not os.path.isdir(run_dir):
        return []
    order = list((meta or {}).get("methods") or [])
    out = []
    for p in glob.glob(os.path.join(run_dir, "*", "seed_*", META)):
        job = os.path.dirname(p)
        m, s = os.path.basename(os.path.dirname(job)), os.path.basename(job)[5:]
        try:
            s = int(s)
        except ValueError:
            continue
        out.append((m, s, job))
    out.sort(key=lambda x: (order.index(x[0]) if x[0] in order else 99, x[0], x[1]))
    if os.path.exists(os.path.join(run_dir, META)):
        out.append(("", "", run_dir))
    return out


def newest_meta(run_dir, meta=None):
    js = jobs(run_dir, meta)
    if not js:
        return None
    job = max(js, key=lambda x: os.path.getmtime(os.path.join(x[2], META)))[2]
    return _load(os.path.join(job, META))


def tls_of(mm):
    """(TLS version, cipher) as negotiated by the drones (first connected drone), else (None, None)."""
    for d in (mm or {}).get("drones") or []:
        if d.get("status", "connected") == "connected" and (d.get("cipher") or d.get("tls_version")):
            return d.get("tls_version"), d.get("cipher")
    return None, None


def _tls_name(v):
    s = str(v or "").strip()
    m = re.match(r"(?i)^tls\s*v?\s*(\d)[._](\d)$", s)  # "TLSv1.3" / "TLS1_3" → "TLS 1.3"
    return f"TLS {m.group(1)}.{m.group(2)}" if m else (s or TLS_DEFAULT)


def link_label(run_dir, meta):
    """Status-bar text for an mTLS run: 'mTLS · TLS 1.3 · <cipher>' (cipher from the newest job's mtls_meta.json)."""
    if not is_mtls(meta):
        return None
    ver, cipher = tls_of(newest_meta(run_dir, meta))
    return " · ".join(x for x in ("mTLS", _tls_name(ver), cipher) if x)


# ---------- formatting ----------
def _when(x):
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))
    return str(x).replace("T", " ")[:19]


def _bytes(b):
    b = float(b or 0)
    for unit, k in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= k:
            return f"{b / k:.2f} {unit}"
    return f"{b:.0f} B"


def _roles(meta):
    """{cid: role} from the run's first seed, only when the role-colour switch allows it on server-side pages."""
    if not app_roles.server_on():
        return {}
    parts = (meta or {}).get("partitions") or {}
    p = parts.get(str(((meta or {}).get("seeds") or [None])[0])) or next(iter(parts.values()), {}) or {}
    return {c.get("cid"): c.get("role") for c in p.get("clients", [])}


def _cid(d, i):
    try:
        return int(d.get("cid", i))
    except (TypeError, ValueError):
        return i


def connection_frame(mm, log=None):
    """Per-drone connection rows from mtls_meta.json (+ run totals of wire bytes from mtls.jsonl)."""
    up, down = {}, {}
    for line in log or []:
        for d in line.get("drones") or []:
            c = d.get("cid")
            up[c] = up.get(c, 0) + (d.get("bytes_up") or 0)
            down[c] = down.get(c, 0) + (d.get("bytes_down") or 0)
    rows = []
    for i, d in enumerate((mm or {}).get("drones") or []):
        c = _cid(d, i)
        row = {"Drone": drone_name(c), "Status": d.get("status", "?"), "TLS": _tls_name(d.get("tls_version"))
               if d.get("tls_version") else "–", "Cipher": d.get("cipher") or "–", "Subject": d.get("subject") or "–",
               "Serial": str(d.get("serial") if d.get("serial") is not None else "–"),
               "SHA-256 fingerprint": d.get("fingerprint") or "–", "Valid until": _when(d.get("not_after")) or "–",
               "Handshake (ms)": d.get("handshake_ms"), "Probe handshake (ms)": d.get("probe_handshake_ms"),
               "ALPN": d.get("alpn") or "–", "Connected at": _when(d.get("connected_at") or d.get("refused_at")) or "–",
               "PID": d.get("pid")}
        if d.get("status") == "refused" or d.get("reason"):
            row["Reason"] = d.get("reason") or "–"
            pres = d.get("presented") or {}
            for k, col in (("subject", "Subject"), ("serial", "Serial"), ("fingerprint", "SHA-256 fingerprint"),
                           ("not_after", "Valid until")):  # the certificate the refused drone presented
                if row[col] in ("–", None) and pres.get(k):
                    row[col] = _when(pres[k]) if k == "not_after" else str(pres[k])
        if log:
            row.update({"Wire up": _bytes(up.get(c, 0)), "Wire down": _bytes(down.get(c, 0))})
        rows.append((c, row))
    return [c for c, _ in rows], pd.DataFrame([r for _, r in rows])


def connection_table(mm, roles=None, log=None):
    cids, df = connection_frame(mm, log)
    if df.empty:
        st.info("No drone connections logged yet.")
        return
    sty = app_roles.rows(df, [(roles or {}).get(c) for c in cids], ["Drone"])
    sty = sty.format({"Handshake (ms)": "{:.2f}", "Probe handshake (ms)": "{:.2f}", "PID": "{:.0f}"}, na_rep="–")
    st.dataframe(sty, hide_index=True, width="stretch",
                 column_config={"SHA-256 fingerprint": st.column_config.TextColumn(width="medium"),
                                "Subject": st.column_config.TextColumn(width="small")})
    refused = [drone_name(c) for c, s in zip(cids, df["Status"]) if s == "refused"]
    if refused:
        st.warning(f"Refused at authentication: {', '.join(refused)} (client certificate revoked, unknown or not "
                   "matching its drone id; reason in the table).")


def endpoint_lines(mm):
    """Markdown lines: transport, endpoint, CA and server certificate, library versions."""
    mm = mm or {}
    ca, srv = mm.get("ca") or {}, mm.get("server") or {}
    out = [f"**Transport**: {mm.get('transport', 'gRPC over TLS 1.3 (mTLS)')}"
           + (f" · server on `127.0.0.1:{mm['port']}`" if mm.get("port") else "")
           + (f" · gRPC {mm['grpc_version']}" if mm.get("grpc_version") else "")
           + (f" ({mm['grpc_tls_library']})" if mm.get("grpc_tls_library") else "")
           + (f" · {mm['openssl']}" if mm.get("openssl") else "")
           + (f" · all drones connected in {mm['connect_s']:.2f} s" if isinstance(mm.get("connect_s"), (int, float)) else "")]
    for name, c in (("CA", ca), ("Server certificate", srv)):
        if c:
            out.append(f"**{name}**: `{c.get('subject', '–')}` · serial `{c.get('serial', '–')}` · SHA-256 "
                       f"`{c.get('fingerprint', '–')}` · " + (f"{c['key']} / {c.get('signature') or '?'} · " if c.get("key") else "")
                       + f"valid until {_when(c.get('not_after')) or '–'}")
    return out


def failure(mm, meta=None):
    """Shows a failed / refused mTLS job clearly (mtls_meta status 'refused', drones' reasons, the run's errors)."""
    bad = [d for d in (mm or {}).get("drones") or [] if d.get("status") == "refused"]
    errs = [e.get("error", "") for e in (meta or {}).get("errors") or [] if "mtls" in str(e.get("error", "")).lower()]
    if (mm or {}).get("status") == "refused" or bad:
        why = "; ".join(f"{drone_name(_cid(d, 0))}: {d.get('reason') or 'refused'}" for d in bad)
        st.error("**This mTLS job stopped at authentication**: the server refused "
                 + (why or "a drone") + ". A revoked or unknown drone certificate makes the job fail instead of "
                 "training without that drone; restore it in Secure Link → Certificates and start a new run.")
    for e in errs:
        st.error(e)


def traffic_frame(log):
    rows = []
    for i, line in enumerate(log or []):
        ds = line.get("drones") or []
        r = int(line.get("round", i)) + 1
        ms_up = [d.get("rpc_ms_up") for d in ds if d.get("rpc_ms_up") is not None]
        ms_dn = [d.get("rpc_ms_down") for d in ds if d.get("rpc_ms_down") is not None]
        for k, lab, ms in (("bytes_up", "Drones → server", ms_up), ("bytes_down", "Server → drones", ms_dn)):
            rows.append({"Round": r, "Direction": lab, "kB": sum(d.get(k) or 0 for d in ds) / 1e3,
                         "RPC ms (mean)": sum(ms) / len(ms) if ms else None, "Server wait ms": line.get("server_wait_ms")})
    return pd.DataFrame(rows)


def traffic_chart(log, height=200):
    df = traffic_frame(log)
    if df.empty:
        st.info("No round logged yet (`mtls.jsonl`).")
        return
    dark = bool(st.session_state.get("ui_dark"))
    cols = ["#4a8cf7", "#9aa4b2"] if dark else ["#1565c0", "#90a4ae"]
    st.altair_chart(alt.Chart(df).mark_line(strokeWidth=1.8, interpolate="step-after").encode(
        x=alt.X("Round:Q", title="FL round"), y=alt.Y("kB:Q", title="Wire bytes per round (kB, gRPC messages)"),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=["Drones → server", "Server → drones"], range=cols),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Round", "Direction", alt.Tooltip("kB:Q", format=",.1f"), alt.Tooltip("RPC ms (mean):Q", format=".1f"),
                 alt.Tooltip("Server wait ms:Q", format=".1f")]).properties(height=height), width="stretch")


def _metrics(mm, log):
    ds = (mm or {}).get("drones") or []
    ok = sum(1 for d in ds if d.get("status") == "connected")
    hs = [d.get("handshake_ms") for d in ds if isinstance(d.get("handshake_ms"), (int, float))]
    up = sum(d.get("bytes_up") or 0 for x in log for d in x.get("drones") or [])
    dn = sum(d.get("bytes_down") or 0 for x in log for d in x.get("drones") or [])
    wait = [x.get("server_wait_ms") for x in log if isinstance(x.get("server_wait_ms"), (int, float))]
    m = st.columns(5)
    m[0].metric("Drones connected", f"{ok}/{len(ds)}" if ds else "–",
                help="Mutual TLS: each drone process presented its own X.509 client certificate and the server checked "
                     "its CN against the drone id and the revocation list.")
    m[1].metric("Handshake (mean)", f"{sum(hs) / len(hs):.1f} ms" if hs else "–")
    m[2].metric("Rounds on the wire", len(log))
    m[3].metric("Wire traffic", _bytes(up + dn), help="gRPC message sizes (drones → server + server → drones); TLS "
                                                       "records add about 22-29 bytes each on top.")
    m[4].metric("Server wait / round", f"{sum(wait) / len(wait) / 1000:.2f} s" if wait else "–",
                help="Time the server waited for the drones' results each round (mtls.jsonl server_wait_ms).")


# ---------- Secure Link → This run's link ----------
def render_run_panel(run_dir, meta):
    """Real mTLS section of 'This run's link' (below the link animation): connections + wire traffic per round."""
    js = jobs(run_dir, meta)
    if not js:
        if is_mtls(meta):
            st.markdown("#### Real mTLS link (gRPC over TLS 1.3)")
            failure(None, meta)
            if not meta.get("errors"):
                st.info("Drones are connecting: the connection table appears once the first TLS handshakes are "
                        "logged (`mtls_meta.json`).")
        return
    run = os.path.basename(os.path.normpath(run_dir))
    st.markdown("#### Real mTLS link (gRPC over TLS 1.3, separate drone processes)")
    # follow the job chosen in the link view above (same method / seed) when it has an mTLS log
    want = (st.session_state.get(f"sl_method_{run}"), st.session_state.get(f"sl_seed_{run}"))
    hit = next((i for i, (m, s, _) in enumerate(js) if (m, s) == want), None)
    if hit is None and want[0] and os.path.isdir(os.path.join(run_dir, want[0], f"seed_{want[1]}")):
        st.caption(f"The job chosen above ({want[0].replace('_', ' ')} · seed {want[1]}) has no mTLS files (e.g. "
                   "Fed-GraB trains in-process); showing another job of this run.")
    if hit is None and len(js) > 1:
        lab = lambda i: f"{js[i][0].replace('_', ' ')} · seed {js[i][1]}" if js[i][0] else "this run"
        hit = st.selectbox("mTLS job", range(len(js)), format_func=lab, key=f"sl_mtls_job_{run}")
    m, s, job = js[hit or 0]
    mm = _load(os.path.join(job, META)) or {}
    log = read_log(job)
    if m:
        st.caption(f"Job: {m.replace('_', ' ')} · seed {s} · `{os.path.relpath(job, ROOT)}/{META}`, `{LOG}`"
                   + (f" · link {mm['status']}" if mm.get("status") else ""))
    failure(mm, meta)
    _metrics(mm, log)
    for line in endpoint_lines(mm):
        st.markdown(line)
    connection_table(mm, _roles(meta), log)
    st.markdown("##### Wire traffic per round")
    traffic_chart(log)
    if meta.get("status") in ("running", "preparing"):
        st.caption("A round's wire record is written when the next round starts, so this chart lags one round "
                   "while training.")
    app_roles_note = ("Drone names coloured by true role (simulation view); switch it off in the sidebar to see the "
                      "table as the server does.") if app_roles.server_on() else ""
    st.caption("Connection fields: each drone process's client certificate (subject / serial / fingerprint / expiry), "
               "its handshake time and process id. " + (mm.get("tls_info_source", "").capitalize() + ". "
                                                        if mm.get("tls_info_source") else "")
               + "Wire bytes = gRPC message sizes; the message-level encryption runs on top of TLS. " + app_roles_note)


# ---------- Full demo (mTLS source) ----------
def render_demo_panel(job):
    mm = _load(os.path.join(job, META))
    st.markdown("#### Step 1 with real certificates: mTLS enrollment of this run")
    if not mm:
        st.info(f"`{os.path.relpath(job, ROOT)}/{META}` not found.")
        return
    st.caption("The demo above is built from this run's message-level link log; the drones of this run also talked "
               "to the server over real mutual TLS (gRPC, separate processes). These are the X.509 certificates they "
               "presented and the TLS handshakes they made (`" + os.path.relpath(job, ROOT) + f"/{META}`).")
    for line in endpoint_lines(mm):
        st.markdown(line)
    meta = _load(os.path.join(os.path.dirname(os.path.dirname(job)), "meta.json")) or {}
    failure(mm, meta)
    if mm.get("tls_info_source"):
        st.caption("TLS version / cipher: " + mm["tls_info_source"] + ".")
    log = read_log(job)
    _metrics(mm, log)
    connection_table(mm, _roles(meta), log)
    if log:
        with st.expander("Wire traffic per round (`mtls.jsonl`)"):
            traffic_chart(log, 180)


# ---------- Certificates (project PKI) ----------
# The PKI belongs to the transport (fldos/mtls.py): this view creates, reads and revokes through its functions
# (ensure_project_pki, pki_info, set_revoked, load_revoked) and never writes revoked.json itself. Without that module
# (older code) the certificates are only listed, read-only, by the small PEM reader below.
N_DRONES = 10


def backend():
    """fldos.mtls when this code version has the mTLS transport, else None."""
    try:
        from fldos import mtls
    except Exception:  # noqa: BLE001 - transport module absent or failing to import: read-only view
        return None
    need = ("ensure_project_pki", "pki_info", "set_revoked", "load_revoked")
    return mtls if all(hasattr(mtls, f) for f in need) else None


def _hexs(x):
    """Serial as the transport writes it: lower-case hex without leading zeros."""
    if isinstance(x, int):
        return format(x, "x")
    return str(x or "").lower().replace(":", "").lstrip("0") or "0"


def _cn_cid(cn):
    m = re.match(r"(?i)^drone[-_ ]?(\d+)$", str(cn or ""))
    return int(m.group(1)) - 1 if m else None


def _row(info, kind, cid=None, revoked=False):
    return {"kind": kind, "cid": cid, "cn": info.get("cn"), "subject": info.get("subject"), "issuer": info.get("issuer"),
            "serial": _hexs(info.get("serial")), "fingerprint": info.get("fingerprint"),
            "not_before": info.get("not_before"), "not_after": info.get("not_after"), "key": info.get("key"),
            "signature": info.get("signature"), "revoked": bool(revoked)}


def pki_rows(pki_dir, n=N_DRONES):
    """[cert row] for CA, server, drones: from fldos.mtls.pki_info when available, else from the PEM files."""
    mt = backend()
    if mt is not None:
        info = mt.pki_info(pki_dir, n)
        rows = [_row(info[k], k if k != "ca" else "CA") for k in ("ca", "server") if info.get(k)]
        rows += [_row(d, "drone", d.get("cid"), d.get("revoked")) for d in info.get("drones") or []]
        return rows
    rev = _fallback_revoked(pki_dir)
    rows = []
    for p in sorted(glob.glob(os.path.join(pki_dir, "*.pem"))):
        try:
            info = _pem_info(open(p, "rb").read())
        except Exception:  # noqa: BLE001 - not a certificate
            continue
        cid = _cn_cid(info["cn"])
        kind = "CA" if info["is_ca"] else ("drone" if cid is not None else "server")
        rows.append(_row(info, kind, cid, _hexs(info["serial"]) in rev))
    order = {"CA": 0, "server": 1, "drone": 2}
    return sorted(rows, key=lambda r: (order[r["kind"]], r["cid"] if r["cid"] is not None else 0))


def _pem_info(data):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.x509.oid import NameOID
    c = x509.load_pem_x509_certificate(data)
    cn = c.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    try:
        is_ca = c.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    except x509.ExtensionNotFound:
        is_ca = c.subject == c.issuer
    pub = c.public_key()
    return {"subject": c.subject.rfc4514_string(), "issuer": c.issuer.rfc4514_string(), "cn": cn[0].value if cn else None,
            "serial": f"{c.serial_number:x}", "fingerprint": c.fingerprint(hashes.SHA256()).hex(),
            "not_before": c.not_valid_before_utc.isoformat(), "not_after": c.not_valid_after_utc.isoformat(),
            "key": f"ECDSA {pub.curve.name}" if hasattr(pub, "curve") else type(pub).__name__,
            "signature": c.signature_hash_algorithm.name.upper() if c.signature_hash_algorithm else None, "is_ca": is_ca}


def _fallback_revoked(pki_dir):
    v = _load(os.path.join(pki_dir, REVOKED))
    v = v.get("revoked", []) if isinstance(v, dict) else (v or [])
    return {_hexs(x) for x in v if isinstance(x, (int, str))}


def _ask(cid, action):
    st.session_state["pki_pending"] = {"cid": cid, "action": action}


def _cancel():
    st.session_state.pop("pki_pending", None)


def _confirm(cid, action, pki_dir):
    mt = backend()
    try:
        mt.set_revoked(pki_dir, cid, revoked=action == "revoke")
        st.session_state["pki_flash"] = (f"{drone_name(cid)} {'revoked' if action == 'revoke' else 'restored'}; the next "
                                         "mTLS run " + ("refuses it at authentication." if action == "revoke"
                                                        else "accepts it again."))
    except Exception as e:  # noqa: BLE001 - shown to the operator
        st.session_state["pki_flash"] = f"ERROR: could not {action} {drone_name(cid)}: {e}"
    st.session_state.pop("pki_pending", None)


def _create_project_pki():
    try:
        backend().ensure_project_pki(ROOT, N_DRONES)
        st.session_state["pki_flash"] = ("Project PKI created in `pki/`: every mTLS run from now on uses these "
                                         "certificates and this revocation list.")
    except Exception as e:  # noqa: BLE001 - shown to the operator
        st.session_state["pki_flash"] = f"ERROR: could not create the project PKI: {e}"


def _day(iso):
    return str(iso or "")[:10]


def cert_table(rows, roles, where):
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    def status(c):
        if c["revoked"]:
            return "revoked"
        if c["not_after"] and c["not_after"] < now:
            return "expired"
        return "not yet valid" if c["not_before"] and c["not_before"] > now else "valid"

    names = [{"CA": "Certificate authority", "server": "Aggregation server"}.get(c["kind"]) or drone_name(c["cid"])
             for c in rows]
    df = pd.DataFrame([{"Certificate": n, "Status": status(c), "Subject": c["subject"], "Issuer": c["issuer"],
                        "Serial (hex)": c["serial"], "SHA-256 fingerprint": c["fingerprint"],
                        "Valid from": _day(c["not_before"]), "Valid until": _day(c["not_after"]), "Key": c["key"] or "–",
                        "Signature": c["signature"] or "–"} for n, c in zip(names, rows)])
    st.dataframe(app_roles.rows(df, [roles.get(c["cid"]) if c["kind"] == "drone" else None for c in rows],
                                ["Certificate"]), hide_index=True, width="stretch")
    st.caption(f"{len(rows)} certificates in `{where}/` · {sum(c['revoked'] for c in rows)} revoked (`{REVOKED}`). "
               "Private keys (`*.key`) are never shown.")


def render_certificates(roles=None, pki_dir=PKI_DIR, run_pkis=()):
    """Project PKI (pki/) with Revoke / Restore per drone. run_pkis: [(label, dir)] of runs' own PKIs, listed read-only
    while no project PKI exists (each such run made a throwaway PKI in its own folder)."""
    st.markdown("#### Project PKI: certificate authority and certificates")
    roles, mt, where = roles or {}, backend(), os.path.relpath(pki_dir, ROOT)
    flash = st.session_state.pop("pki_flash", None)
    if flash:
        (st.error if flash.startswith("ERROR") else st.success)(flash)
    if not os.path.isdir(pki_dir):
        st.info(f"No project PKI yet (`{where}/`). Until it exists, each Real mTLS run makes its own throwaway PKI in "
                "its run folder (a certificate authority, the server certificate and one client certificate per drone, "
                "CN drone-01 … drone-10), so a revocation would not carry over to the next run. Once the project PKI "
                "exists, every mTLS run uses it, and it is listed here with Revoke / Restore per drone.")
        if mt is not None:
            st.button("Create the project PKI", key="pki_create", on_click=_create_project_pki,
                      help="Creates pki/ (CA, server and 10 drone certificates, empty revoked.json) with the transport's "
                           "own code (fldos/mtls.py ensure_project_pki). Existing files are never overwritten.")
        for label, d in run_pkis:
            if os.path.isdir(d):
                try:
                    rows = pki_rows(d)
                except Exception:  # noqa: BLE001 - unreadable run PKI: skip it
                    continue
                if rows:
                    st.markdown(f"##### Certificates of {label} (its own PKI, read-only)")
                    cert_table(rows, roles, os.path.relpath(d, ROOT))
        return
    try:
        rows = pki_rows(pki_dir)
    except Exception as e:  # noqa: BLE001 - unreadable / unexpected files: say so, keep the console usable
        st.error(f"Could not read the certificates in `{where}/`: {e}")
        return
    if not rows:
        st.info(f"`{where}/` holds no certificates yet.")
        return
    cert_table(rows, roles, where)
    st.caption("Revocation is checked by the server at each drone's authentication step: at the next mTLS run a "
               "revoked drone is refused (UNAUTHENTICATED; its connection shows 'refused') and that training job "
               "stops with 'refused at authentication' instead of training without it.")
    drones = [c for c in rows if c["kind"] == "drone"]
    if not drones:
        return
    st.markdown("##### Revoke / restore a drone's certificate")
    if mt is None:
        st.caption("Read-only here: this code version has no mTLS transport module (fldos/mtls.py) to revoke with.")
        return
    pend = st.session_state.get("pki_pending")
    if pend:
        cert = next((c for c in drones if c["cid"] == pend.get("cid")), None)
        if cert is None:
            _cancel()
        else:
            act = pend["action"]
            with st.container(border=True):
                st.markdown(f"**{'Revoke' if act == 'revoke' else 'Restore'} "
                            f"{app_roles.span(drone_name(cert['cid']), roles.get(cert['cid']))}** "
                            f"(`{cert['subject']}`, serial `{cert['serial']}`)?", unsafe_allow_html=True)
                st.caption(f"Updates `{where}/{REVOKED}` (fldos/mtls.py set_revoked). " + (
                    "The server refuses this drone's certificate from the next mTLS run on (that job stops at "
                    "authentication)." if act == "revoke" else "The drone's certificate is accepted again from the "
                                                                 "next mTLS run."))
                b = st.columns([1, 1, 4])
                b[0].button("Confirm", type="primary", key="pki_yes", on_click=_confirm,
                            args=(cert["cid"], act, pki_dir), width="stretch")
                b[1].button("Cancel", key="pki_no", on_click=_cancel, width="stretch")
    for c in drones:
        r = st.columns([2, 2, 2, 1.4])
        r[0].markdown(app_roles.span(drone_name(c["cid"]), roles.get(c["cid"]), bold=True) + f" · `{c['cn']}`",
                      unsafe_allow_html=True)
        r[1].markdown(f"serial `{c['serial'][:16]}…`" if len(c["serial"]) > 16 else f"serial `{c['serial']}`")
        r[2].markdown(":orange[**revoked**]" if c["revoked"] else f"valid until {_day(c['not_after'])}")
        act = "restore" if c["revoked"] else "revoke"
        r[3].button(act.capitalize(), key=f"pki_{act}_{c['cid']}", on_click=_ask, args=(c["cid"], act),
                    width="stretch", disabled=bool(pend))
