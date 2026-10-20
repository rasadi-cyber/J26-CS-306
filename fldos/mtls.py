"""Opt-in real mTLS transport (security.transport = "mtls"; default "inproc"): the aggregation server and the 10 drones
are separate processes that talk gRPC over TLS 1.3 with mutual X.509 authentication (notes/SPEC_P18_MTLS.md).

PKI (`cryptography` X.509, ECDSA P-256 + SHA-256)
  CA (self-signed, 1 year), server certificate (SAN 127.0.0.1 + localhost, serverAuth), one client certificate per
  drone (CN drone-01 .. drone-NN, clientAuth). Files in the PKI folder:
    ca.pem ca.key  server.pem server.key  drone-01.pem drone-01.key ...  revoked.json
  revoked.json = JSON list of revoked certificate serials (lower-case hex strings; ints are accepted too).
  Folder: security.pki_dir if set (relative to the project root), else the persistent project PKI <root>/pki when it
  exists (create it with ensure_project_pki), else <run_dir>/pki (one throwaway PKI per run).
Server (inside the job process)
  grpc server on 127.0.0.1:<free port>, ssl_server_credentials(require_client_auth=True): the TLS handshake already
  rejects certificates not issued by the CA. On every call the server also reads the caller's certificate from
  context.auth_context(): its CN must be drone-XX of the drone id in the request and its serial must not be in
  revoked.json, otherwise the call is aborted with UNAUTHENTICATED (status "refused" in mtls_meta.json) and the job
  stops with an error (an update is never silently dropped).
  Two generic bytes methods (no .proto): /fldos.Swarm/Next (a drone asks for its next task; long poll: the server
  answers train | score | stop, or idle after IDLE_S) and /fldos.Swarm/Done (a drone returns a result).
  Message = 4-byte little-endian header length + JSON header + payload (float32 little-endian model vector; a train
  result adds the drone's torch RNG state bytes).
Drones (spawned children of the job process; same process group, so the dashboard's Stop kills them)
  Each holds a FedSim replica built from the shared memory-mapped partition exactly like fldos/parallel.py workers,
  connects with its own certificate and only ever trains / scores its own drone id. A drone exits when the server
  says stop, when its parent process is gone, or when the server has been unreachable for more than 10 s.
Identity (bit-identical to the in-process path), as fldos/parallel.DronePool: training is seeded per (seed, round,
  drone) inside FedSim.local_train, results are returned in drone order, the parent's torch RNG is set to the state of
  the last drone trained, and model vectors cross the wire as their exact float32 bytes.
TLS version and cipher: grpcio's Python API does not expose the negotiated TLS version / cipher suite (auth_context()
  only carries the certificate, transport_security_type and security_level). Each drone therefore opens one extra
  `ssl` test connection to the same server port with its own certificate (minimum_version = TLS 1.3, ALPN h2) and
  records ssl.SSLSocket.version() and .cipher(): the values the gRPC server negotiates with that client certificate.
Logs per job (only with transport mtls): mtls_meta.json, mtls.jsonl (one line per round), live_state.json
  ("connecting" drones until their first authenticated call).
"""
import datetime
import fcntl
import hashlib
import json
import os
import queue
import socket
import ssl
import struct
import sys
import threading
import time
import traceback
from concurrent import futures

import torch

from .securelink import bytes_vec, vec_bytes

# Drones are started with spawn (fork + exec), so gRPC's fork handlers are not needed; without them gRPC also stops
# logging "skipping fork() handlers" each time a job starts its drones. Set before grpc is first imported.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "false")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

SERVICE = "fldos.Swarm"
IDLE_S = 5.0            # long-poll: an idle drone gets {"op": "idle"} after this and asks again
UNREACHABLE_S = 10.0    # a drone exits when the server has been unreachable for longer than this
CONNECT_TIMEOUT_S = 240.0
MAX_MSG = 256 * 1024 * 1024
TRANSPORT = "gRPC over TLS 1.3 (mTLS)"
ORG = "FL DoS Swarm"


def enabled(cfg):
    return cfg.get("security", {}).get("transport", "inproc") == "mtls"


def drone_cn(cid):
    return f"drone-{cid + 1:02d}"


# ---------------------------------------------------------------- PKI
def _x509():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    return x509, hashes, serialization, ec, ExtendedKeyUsageOID, NameOID


def _write(path, data, private=False):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    if private:
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _new_key():
    _, _, _, ec, _, _ = _x509()
    return ec.generate_private_key(ec.SECP256R1())


def _key_pem(key):
    _, _, serialization, _, _, _ = _x509()
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _issue(cn, key, ca_cert=None, ca_key=None, kind="client", days=365):
    """X.509 certificate for key: self-signed CA (kind ca), server (SAN 127.0.0.1 + localhost) or drone client."""
    import ipaddress
    x509, hashes, serialization, ec, EKU, NameOID = _x509()
    name = x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORG), x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
    b = (x509.CertificateBuilder().subject_name(name).issuer_name(ca_cert.subject if ca_cert is not None else name)
         .public_key(key.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=days))
         .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False))
    if kind == "ca":
        b = (b.add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
             .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                          content_commitment=False, key_encipherment=False, data_encipherment=False,
                                          key_agreement=False, encipher_only=False, decipher_only=False), critical=True))
        return b.sign(key, hashes.SHA256())
    b = (b.add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=False, crl_sign=False,
                                      content_commitment=False, key_encipherment=False, data_encipherment=False,
                                      key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
         .add_extension(x509.ExtendedKeyUsage([EKU.SERVER_AUTH if kind == "server" else EKU.CLIENT_AUTH]),
                        critical=False))
    if kind == "server":
        b = b.add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                                                         x509.DNSName("localhost")]), critical=False)
    return b.sign(ca_key, hashes.SHA256())


def _cert_pem(cert):
    _, _, serialization, _, _, _ = _x509()
    return cert.public_bytes(serialization.Encoding.PEM)


def cert_info(cert_or_pem):
    """Displayed fields of a certificate (PEM bytes or a cryptography Certificate)."""
    x509, hashes, _, _, _, NameOID = _x509()
    c = cert_or_pem
    if isinstance(c, str):
        c = c.encode()
    if isinstance(c, bytes):
        c = x509.load_pem_x509_certificate(c)
    cn = c.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    pub = c.public_key()
    return {"subject": c.subject.rfc4514_string(), "issuer": c.issuer.rfc4514_string(),
            "cn": cn[0].value if cn else None, "serial": f"{c.serial_number:x}",
            "fingerprint": c.fingerprint(hashes.SHA256()).hex(),
            "not_before": c.not_valid_before_utc.isoformat(), "not_after": c.not_valid_after_utc.isoformat(),
            "key": f"ECDSA {pub.curve.name}" if hasattr(pub, "curve") else type(pub).__name__,
            "signature": f"{c.signature_hash_algorithm.name.upper()}"
            if c.signature_hash_algorithm is not None else None}


def pki_paths(pki_dir, n=10):
    p = lambda f: os.path.join(pki_dir, f)
    return {"ca": p("ca.pem"), "ca_key": p("ca.key"), "server": p("server.pem"), "server_key": p("server.key"),
            "revoked": p("revoked.json"),
            "drones": {c: {"cert": p(f"{drone_cn(c)}.pem"), "key": p(f"{drone_cn(c)}.key")} for c in range(n)}}


def ensure_pki(pki_dir, n=10):
    """Create (or complete) a PKI in pki_dir: CA, server certificate, n drone certificates, revoked.json.
    Existing files are kept (so a persistent PKI keeps its certificates and revocations). -> pki_paths()."""
    x509, _, serialization, _, _, _ = _x509()
    os.makedirs(pki_dir, exist_ok=True)
    P = pki_paths(pki_dir, n)
    with open(os.path.join(pki_dir, ".lock"), "w") as lk:  # concurrent jobs of one run share the folder
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not (os.path.exists(P["ca"]) and os.path.exists(P["ca_key"])):
            ca_key = _new_key()
            ca = _issue(f"{ORG} Operator CA", ca_key, kind="ca")
            _write(P["ca_key"], _key_pem(ca_key), private=True)
            _write(P["ca"], _cert_pem(ca))
        ca = x509.load_pem_x509_certificate(open(P["ca"], "rb").read())
        ca_key = serialization.load_pem_private_key(open(P["ca_key"], "rb").read(), None)
        todo = [("aggregation-server", P["server"], P["server_key"], "server")] + \
               [(drone_cn(c), d["cert"], d["key"], "client") for c, d in P["drones"].items()]
        for cn, cp, kp, kind in todo:
            if os.path.exists(cp) and os.path.exists(kp):
                continue
            key = _new_key()
            _write(kp, _key_pem(key), private=True)
            _write(cp, _cert_pem(_issue(cn, key, ca, ca_key, kind)))
        if not os.path.exists(P["revoked"]):
            _write(P["revoked"], b"[]\n")
    return P


def ensure_project_pki(root, n=10):
    """The persistent project PKI <root>/pki (used by every mTLS run while it exists; revocations carry over)."""
    return ensure_pki(os.path.join(root, "pki"), n)


def pki_dir_for(cfg, root, run_dir):
    sc = cfg.get("security", {})
    if sc.get("pki_dir"):
        return os.path.join(root, sc["pki_dir"])
    if os.path.isdir(os.path.join(root, "pki")):
        return os.path.join(root, "pki")
    return os.path.join(run_dir, "pki")


def _serial(s):
    return f"{s:x}" if isinstance(s, int) else str(s).lower().replace(":", "").lstrip("0") or "0"


def load_revoked(pki_dir):
    p = os.path.join(pki_dir, "revoked.json")
    if not os.path.exists(p):
        return set()
    v = json.load(open(p))
    v = v.get("revoked", []) if isinstance(v, dict) else v
    return {_serial(s) for s in v}


def set_revoked(pki_dir, cid, revoked=True):
    """Revoke (or restore) drone cid's certificate: adds / removes its serial in revoked.json. -> list of serials."""
    info = cert_info(open(pki_paths(pki_dir, cid + 1)["drones"][cid]["cert"], "rb").read())
    cur = sorted(load_revoked(pki_dir) - {info["serial"]} | ({info["serial"]} if revoked else set()))
    _write(os.path.join(pki_dir, "revoked.json"), (json.dumps(cur, indent=2) + "\n").encode())
    return cur


def pki_info(pki_dir, n=10):
    """CA + server + drone certificate fields of a PKI folder, with each drone's revoked flag (for the UI)."""
    P = pki_paths(pki_dir, n)
    rev = load_revoked(pki_dir)
    rd = lambda f: cert_info(open(f, "rb").read()) if os.path.exists(f) else None
    drones = []
    for c, d in P["drones"].items():
        i = rd(d["cert"])
        if i is not None:
            drones.append({"cid": c, **i, "revoked": i["serial"] in rev})
    return {"dir": pki_dir, "ca": rd(P["ca"]), "server": rd(P["server"]), "drones": drones, "revoked": sorted(rev)}


# ---------------------------------------------------------------- wire format
def pack(header, payload=b""):
    h = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(h)) + h + payload


def unpack(msg):
    n = struct.unpack_from("<I", msg, 0)[0]
    return json.loads(msg[4:4 + n]), msg[4 + n:]


def _rng_bytes():
    return torch.get_rng_state().numpy().tobytes()


def _bytes_rng(b):
    return torch.frombuffer(bytearray(b), dtype=torch.uint8).clone()


# ---------------------------------------------------------------- drone process
def tls_probe(addr, ca_pem_path, cert_path, key_path):
    """One extra TLS connection to the gRPC server port with the drone's certificate: the TLS version and cipher suite
    the server negotiates (gRPC's Python API does not expose them). TLS 1.3 is required (minimum_version)."""
    host, port = addr.rsplit(":", 1)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_verify_locations(ca_pem_path)
    ctx.load_cert_chain(cert_path, key_path)
    ctx.set_alpn_protocols(["h2"])
    t = time.perf_counter()
    with socket.create_connection((host, int(port)), timeout=10) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as s:
            ms = (time.perf_counter() - t) * 1000
            name, proto, bits = s.cipher()
            return {"tls_version": s.version(), "cipher": name, "cipher_bits": bits, "alpn": s.selected_alpn_protocol(),
                    "probe_handshake_ms": round(ms, 3), "probe_openssl": ssl.OPENSSL_VERSION}


def _drone_main(cfg, seed, shared, method, cid, addr, files, parent_pid, material=None):
    import grpc
    from . import parallel
    from . import securelink as sl

    def watchdog():  # the job process is gone (killed) -> leave at once
        while True:
            if os.getppid() != parent_pid:
                os._exit(0)
            time.sleep(0.5)
    threading.Thread(target=watchdog, daemon=True).start()
    parallel._init(cfg, seed, shared, method)  # FedSim replica from the shared memory-mapped partition
    sim = parallel._SIM
    try:
        probe = tls_probe(addr, files["ca"], files["cert"], files["key"])
    except Exception as e:  # e.g. refused at the TLS handshake: the gRPC call below reports it
        probe = {"tls_version": None, "cipher": None, "probe_error": f"{type(e).__name__}: {e}"}
    rd = lambda p: open(p, "rb").read()
    creds = grpc.ssl_channel_credentials(root_certificates=rd(files["ca"]), private_key=rd(files["key"]),
                                         certificate_chain=rd(files["cert"]))
    opts = [("grpc.max_send_message_length", MAX_MSG), ("grpc.max_receive_message_length", MAX_MSG)]
    ch = grpc.secure_channel(addr, creds, options=opts)
    t = time.perf_counter()
    try:
        grpc.channel_ready_future(ch).result(timeout=UNREACHABLE_S)
    except grpc.FutureTimeoutError:
        print(f"[mtls] {drone_cn(cid)}: no TLS connection to {addr} within {UNREACHABLE_S:.0f} s", file=sys.stderr)
        os._exit(2)
    hello = {"pid": os.getpid(), "handshake_ms": round((time.perf_counter() - t) * 1000, 3),
             "connected_at": time.time(), **probe}
    # message layer (security.secure_link on top of TLS): this drone's own keys, provisioned at enrollment; the drone
    # opens what the server sends it and seals what it sends, so no other process ever holds its private keys
    mat = material
    if mat:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
        skey = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(mat["sign"]))
        spub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(mat["server_sign_pub"]))
        mkeys = {d: bytes.fromhex(k) for d, k in mat["keys"].items()}
    g_cur, last = None, {}  # the global model this drone last received; its last plaintext per direction

    def nack(direction, rnd, error):  # drone -> server re-send request, signed + encrypted by this drone
        body = json.dumps({"type": "nack", "for": direction, "round": rnd, "error": error},
                          sort_keys=True, separators=(",", ":")).encode()
        return sl.seal(mkeys["ctrl"], skey, {"cid": cid, "round": rnd, "dir": "ctrl"}, body)[0]

    def sealed(direction, rnd, plain):
        last[direction] = (plain, rnd)
        t = time.perf_counter()
        pk = sl.seal(mkeys[direction], skey, {"cid": cid, "round": rnd, "dir": direction}, plain)[0]
        return pk, (time.perf_counter() - t) * 1000

    def opened(direction, packet):  # -> (plain, ms) or (None, ms, error)
        t = time.perf_counter()
        try:
            _, plain = sl.open_packet(mkeys[direction], spub, packet)
            return plain, (time.perf_counter() - t) * 1000, None
        except Exception as e:
            return None, (time.perf_counter() - t) * 1000, type(e).__name__ if type(e).__name__ in ("InvalidTag", "InvalidSignature") else "ParseError"

    nxt = ch.unary_unary(f"/{SERVICE}/Next")
    done = ch.unary_unary(f"/{SERVICE}/Done")
    reports, last_ok = [], time.time()

    def call(fn, msg, timeout):
        nonlocal last_ok
        while True:
            try:
                r = fn(msg, timeout=timeout)
                last_ok = time.time()
                return r
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                    print(f"[mtls] {drone_cn(cid)} refused by the server: {e.details()}", file=sys.stderr)
                    os._exit(3)
                if time.time() - last_ok > UNREACHABLE_S:
                    os._exit(2)
                time.sleep(0.2)

    while True:
        req = {"cid": cid, "reports": reports}
        if hello is not None:
            req["hello"] = hello
        t = time.perf_counter()
        resp = call(nxt, pack(req), IDLE_S + UNREACHABLE_S)
        down_ms = (time.perf_counter() - t) * 1000
        hello, reports = None, []
        hdr, payload = unpack(resp)
        op = hdr["op"]
        if op == "stop":
            break
        if op == "idle":
            continue
        down_ms = max(0.0, down_ms - hdr.get("hold_ms", 0.0))
        out = {"cid": cid, "task": hdr["task"], "op": op}
        body = b""
        try:
            if op == "deliver":  # a sealed global model from the server: open + verify it here, keep it
                plain, out["open_ms"], err = opened("downlink", payload)
                out["ok"] = err is None
                if err is None:
                    g_cur = plain
                else:
                    body = nack("downlink", hdr["round"], err)
            elif op == "reseal":  # the server's re-send request (verified first), then the same bytes, fresh nonce
                _, _, err = opened("ctrl", payload)
                if err is not None:
                    raise RuntimeError(f"re-send request failed its checks ({err})")
                plain, r = last[hdr["dir"]]
                body, out["seal_ms"] = sealed(hdr["dir"], r, plain)
            elif op == "train" and mat:  # train from the global this drone received, seal its own update
                from .runner import poison_update
                g = bytes_vec(g_cur)
                t1 = time.perf_counter()
                vec = sim.local_train(g, cid, hdr["round"])
                out["ms"] = (time.perf_counter() - t1) * 1000
                vec = poison_update(cfg, sim.data.clients[cid].role, g, vec)  # a poisoned drone's own behaviour
                body, out["seal_ms"] = sealed("uplink", hdr["round"], vec_bytes(vec))
                out["pkt_len"] = len(body)
                body = body + _rng_bytes()
            elif op == "train":
                t1 = time.perf_counter()
                vec = sim.local_train(bytes_vec(payload), cid, hdr["round"])
                out["ms"] = (time.perf_counter() - t1) * 1000
                vb = vec_bytes(vec)
                out["vec_len"] = len(vb)
                body = vb + _rng_bytes()
            elif op == "score" and hdr.get("sealed"):  # vote: open the model, score it here, seal the ballot
                plain, out["open_ms"], err = opened("vote", payload)
                out["ok"] = err is None
                if err is None:
                    s = getattr(sim, hdr["kind"])(bytes_vec(plain), cid)
                    out["tuple"] = isinstance(s, tuple)
                    body, out["seal_ms"] = sealed("ballot", hdr["round"], sl.score_bytes(s))
                else:
                    body = nack("vote", hdr["round"], err)
            elif op == "score":
                s = getattr(sim, hdr["kind"])(bytes_vec(payload), cid)
                out.update(score=list(s) if isinstance(s, tuple) else [s], tuple=isinstance(s, tuple))
        except Exception:
            out["error"] = traceback.format_exc()
        t = time.perf_counter()
        call(done, pack(out, body), UNREACHABLE_S)
        reports.append({"task": hdr["task"], "round": hdr.get("round"), "down_ms": round(down_ms, 3),
                        "up_ms": round((time.perf_counter() - t) * 1000, 3)})
    ch.close()
    os._exit(0)


# ---------------------------------------------------------------- pool used by runner.run_job
def _write_json(path, obj, indent=2):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


class MTLSPool:
    """fldos/parallel.DronePool interface (train / scores / close) over gRPC + mTLS, one drone process per drone id.

    out_dir: job folder (mtls_meta.json, mtls.jsonl, live_state.json while connecting when own_live); None = no files."""

    def __init__(self, cfg, seed, shared, method, cids, pki_dir, out_dir=None, log=print, own_live=True, material=None):
        import grpc
        import multiprocessing as mp
        self.cids, self.out, self.log, self.own_live = list(cids), out_dir, log, own_live
        self.pki_dir = pki_dir
        n = max(self.cids) + 1
        self.P = ensure_pki(pki_dir, n)
        self.revoked = load_revoked(pki_dir)
        rd = lambda p: open(p, "rb").read()
        self.lock = threading.Lock()
        self.queues = {c: queue.Queue() for c in self.cids}
        self.results = queue.Queue()
        self.hello, self.refused, self._certs = {}, {}, {}
        self.task_id, self.rnd = 0, None
        self.acc, self.wait_ms, self.awaiting = {}, {}, {}  # per round: per-drone traffic, server wait, reports due
        self.flushed = set()
        self.closed = False
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2 * len(self.cids) + 8),
                                  options=[("grpc.max_send_message_length", MAX_MSG),
                                           ("grpc.max_receive_message_length", MAX_MSG)])
        self.server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE, {
            "Next": grpc.unary_unary_rpc_method_handler(self._next),
            "Done": grpc.unary_unary_rpc_method_handler(self._done)}),))
        creds = grpc.ssl_server_credentials([(rd(self.P["server_key"]), rd(self.P["server"]))],
                                            root_certificates=rd(self.P["ca"]), require_client_auth=True)
        self.port = self.server.add_secure_port("127.0.0.1:0", creds)
        self.addr = f"127.0.0.1:{self.port}"
        self.server.start()
        self.meta = {"transport": TRANSPORT, "port": self.port, "address": self.addr, "pki_dir": os.path.abspath(pki_dir),
                     "ca": cert_info(rd(self.P["ca"])), "server": cert_info(rd(self.P["server"])),
                     "drones": [], "revoked": sorted(self.revoked), "grpc_version": grpc.__version__,
                     "grpc_tls_library": "BoringSSL (bundled with grpcio)", "openssl": ssl.OPENSSL_VERSION,
                     "tls_info_source": "per drone: one extra ssl test connection to the gRPC server port with the "
                                        "drone's certificate (minimum TLS 1.3, ALPN h2); grpcio does not expose the "
                                        "negotiated version / cipher", "status": "connecting", "started": time.time()}
        if self.out:
            open(os.path.join(self.out, "mtls.jsonl"), "w").close()
        self.live = {"round": 0, "phase": "connecting", "retransmits": 0, "transport": "mtls",
                     "drones": [{"cid": c, "state": "connecting", "t": time.time()} for c in self.cids]}
        self._live_write()
        ctx = mp.get_context("spawn")
        self.procs = {}
        for c in self.cids:
            files = {"ca": self.P["ca"], **self.P["drones"][c]}
            p = ctx.Process(target=_drone_main, args=(cfg, seed, shared, method, c, self.addr, files, os.getpid(),
                                                      material(c) if material else None),
                            daemon=True, name=f"mtls-{drone_cn(c)}")
            p.start()
            self.procs[c] = p
        try:
            self._wait_connected()
        except BaseException:
            self.close(error=True)
            raise

    # --- bookkeeping
    def _live_write(self):
        if self.out:
            _write_json(os.path.join(self.out, "live_state.json"), self.live, indent=None)

    def _write_meta(self):
        if self.out:
            _write_json(os.path.join(self.out, "mtls_meta.json"), self.meta)

    def _acc(self, cid):
        r = self.acc.setdefault(self.rnd if self.rnd is not None else -1, {})
        return r.setdefault(cid, {"bytes_up": 0, "bytes_down": 0, "rpc_ms_up": 0.0, "rpc_ms_down": 0.0, "tasks": 0})

    def _drone_row(self, cid):
        h = self.hello.get(cid)
        info = self._cert_file_info(cid)
        row = {"cid": cid, **{k: info[k] for k in ("subject", "serial", "fingerprint", "not_after")}}
        if cid in self.refused:
            return {**row, **self.refused[cid], "status": "refused"}
        if h is None:
            return {**row, "status": "connecting", "pid": self.procs[cid].pid if cid in self.procs else None}
        return {**row, "pid": h["pid"], "tls_version": h.get("tls_version"), "cipher": h.get("cipher"),
                "cipher_bits": h.get("cipher_bits"), "alpn": h.get("alpn"), "handshake_ms": h.get("handshake_ms"),
                "probe_handshake_ms": h.get("probe_handshake_ms"),
                "connected_at": h.get("connected_at"), "status": "connected"}

    def _cert_file_info(self, cid):
        if cid not in self._certs:
            self._certs[cid] = cert_info(open(self.P["drones"][cid]["cert"], "rb").read())
        return self._certs[cid]

    # --- gRPC handlers (server threads)
    def _auth(self, context, cid):
        """Caller's certificate (verified against the CA in the TLS handshake): CN must be this drone's, serial not
        revoked; otherwise UNAUTHENTICATED."""
        import grpc
        ac = context.auth_context()
        pem = (ac.get("x509_pem_cert") or [b""])[0]
        info = cert_info(pem) if pem else {"cn": None, "serial": None}
        reason = None
        if info["cn"] != drone_cn(cid):
            reason = f"certificate CN {info['cn']} does not match drone id {drone_cn(cid)}"
        elif info["serial"] in self.revoked:
            reason = f"certificate serial {info['serial']} of {info['cn']} is revoked"
        if reason is None:
            return info
        with self.lock:
            if cid in self.queues and cid not in self.refused:
                self.refused[cid] = {"reason": reason, "presented": {k: info.get(k) for k in
                                     ("subject", "serial", "fingerprint", "not_after")}, "refused_at": time.time()}
        context.abort(grpc.StatusCode.UNAUTHENTICATED, f"refused at authentication: {reason}")

    def _next(self, req, context):
        t0 = time.perf_counter()
        hdr, _ = unpack(req)
        cid = int(hdr["cid"])
        self._auth(context, cid)
        with self.lock:
            self._acc(cid)["bytes_up"] += len(req)
            if "hello" in hdr and cid not in self.hello:
                self.hello[cid] = hdr["hello"]
            for rep in hdr.get("reports", []):
                r = self.acc.setdefault(rep["round"] if rep["round"] is not None else -1, {})
                a = r.setdefault(cid, {"bytes_up": 0, "bytes_down": 0, "rpc_ms_up": 0.0, "rpc_ms_down": 0.0,
                                       "tasks": 0})
                a["rpc_ms_up"] += rep["up_ms"]
                a["rpc_ms_down"] += rep["down_ms"]
                a["tasks"] += 1
                self.awaiting.pop(rep["task"], None)
        try:
            th, payload = self.queues[cid].get(timeout=IDLE_S)
        except queue.Empty:
            th, payload = {"op": "idle"}, b""
        resp = pack({**th, "hold_ms": round((time.perf_counter() - t0) * 1000, 3)}, payload)
        with self.lock:
            self._acc(cid)["bytes_down"] += len(resp)
        return resp

    def _done(self, req, context):
        hdr, payload = unpack(req)
        cid = int(hdr["cid"])
        self._auth(context, cid)
        resp = pack({"ok": True})
        with self.lock:
            a = self._acc(cid)
            a["bytes_up"] += len(req)
            a["bytes_down"] += len(resp)
        self.results.put((cid, hdr, payload))
        return resp

    # --- startup
    def _wait_connected(self):
        t0, shown = time.time(), set()
        while True:
            with self.lock:
                ok, bad = set(self.hello), dict(self.refused)
            for c in sorted((ok | set(bad)) - shown):
                d = self.live["drones"][self.cids.index(c)]
                d.update(state="refused" if c in bad else "connected", t=time.time())
                shown.add(c)
                self._live_write()
            if bad:
                self.meta.update(status="refused", drones=[self._drone_row(c) for c in self.cids])
                self._write_meta()
                why = "; ".join(f"{drone_cn(c)}: {r['reason']}" for c, r in sorted(bad.items()))
                raise RuntimeError(f"mTLS: drone refused at authentication (UNAUTHENTICATED) - {why}")
            if len(ok) == len(self.cids):
                break
            dead = [c for c, p in self.procs.items() if not p.is_alive() and c not in ok]
            if dead:
                time.sleep(0.5)  # a refusal may still be on its way to self.refused
                with self.lock:
                    if self.refused:
                        continue
                raise RuntimeError(f"mTLS: drone process(es) {[drone_cn(c) for c in dead]} exited before connecting "
                                   f"(exit codes {[self.procs[c].exitcode for c in dead]})")
            if time.time() - t0 > CONNECT_TIMEOUT_S:
                raise RuntimeError(f"mTLS: drones not connected after {CONNECT_TIMEOUT_S:.0f} s")
            time.sleep(0.05)
        self.meta.update(status="connected", connect_s=round(time.time() - t0, 3),
                         drones=[self._drone_row(c) for c in self.cids])
        self._write_meta()
        self.live["phase"] = "connected"
        self._live_write()
        self.log(f"[mtls] {len(ok)} drones connected over {TRANSPORT} on {self.addr} "
                 f"({self.meta['drones'][0].get('tls_version')}, {self.meta['drones'][0].get('cipher')})")

    # --- tasks
    def _submit(self, cid, header, payload):
        with self.lock:
            self.task_id += 1
            tid = self.task_id
            self.awaiting[tid] = self.rnd
        self.queues[cid].put(({**header, "task": tid, "round": self.rnd}, payload))
        return tid

    def _collect(self, tasks, on_result=None):
        """Wait for every task's result ({cid: task id}) -> {cid: (header, payload)}; on_result(cid, header) as each
        arrives (completion order)."""
        t0 = time.perf_counter()
        want = {t: c for c, t in tasks.items()}
        got = {}
        while len(got) < len(tasks):
            try:
                cid, hdr, payload = self.results.get(timeout=1.0)
            except queue.Empty:
                dead = [c for c in tasks if not self.procs[c].is_alive()]
                if dead:
                    raise RuntimeError(f"mTLS: drone process(es) {[drone_cn(c) for c in dead]} died "
                                       f"(exit codes {[self.procs[c].exitcode for c in dead]})")
                continue
            if want.get(hdr["task"]) != cid or cid in got:
                continue  # stale or duplicate
            if hdr.get("error"):
                raise RuntimeError(f"mTLS: {drone_cn(cid)} failed:\n{hdr['error']}")
            got[cid] = (hdr, payload)
            if on_result is not None:
                on_result(cid, hdr)
        r = self.rnd if self.rnd is not None else -1
        self.wait_ms[r] = self.wait_ms.get(r, 0.0) + (time.perf_counter() - t0) * 1000
        return got

    def train(self, g, cids, rnd, on_done=None):
        self._flush(before=rnd)
        self.rnd = rnd
        payload = vec_bytes(g)
        tasks = {c: self._submit(c, {"op": "train"}, payload) for c in cids}
        cb = (lambda c, h: on_done(c, h["ms"])) if on_done is not None else None
        got = self._collect(tasks, cb)
        vecs = []
        for c in cids:
            hdr, payload = got[c]
            vecs.append(bytes_vec(payload[:hdr["vec_len"]], g))
        hdr, payload = got[cids[-1]]
        torch.set_rng_state(_bytes_rng(payload[hdr["vec_len"]:]))  # RNG state of the LAST drone, as DronePool
        return vecs

    # --- message layer with real drone processes (securelink.SecureChannel *_remote)
    def _one(self, cid, header, payload):
        return self._collect({cid: self._submit(cid, header, payload)})[cid]

    def train_sealed(self, cids, rnd, on_done=None):
        """Each drone trains from the global it received and returns its update sealed by itself.
        -> {cid: (packet, seal_ms, train_ms)}; torch RNG restored from the last drone (as train())."""
        self._flush(before=rnd)
        self.rnd = rnd
        tasks = {c: self._submit(c, {"op": "train"}, b"") for c in cids}
        got = self._collect(tasks, (lambda c, h: on_done(c, h["ms"])) if on_done is not None else None)
        out = {c: (got[c][1][:got[c][0]["pkt_len"]], got[c][0]["seal_ms"], round(got[c][0]["ms"], 3)) for c in cids}
        hdr, payload = got[cids[-1]]
        torch.set_rng_state(_bytes_rng(payload[hdr["pkt_len"]:]))
        return out

    def deliver(self, cid, packet):
        """A sealed global model to one drone -> (ok, None | the drone's NACK packet, open_ms)."""
        h, body = self._one(cid, {"op": "deliver"}, packet)
        return bool(h.get("ok")), (None if h.get("ok") else body), h.get("open_ms", 0.0)

    def reseal(self, cid, direction, nack):
        """The server's re-send request -> the drone's fresh packet (packet, seal_ms)."""
        h, body = self._one(cid, {"op": "reseal", "dir": direction}, nack)
        return body, h.get("seal_ms", 0.0)

    def score_sealed(self, kind, cid, packet):
        """A sealed model to a voter -> (True, (ballot packet, seal_ms, tuple?), open_ms) or (False, NACK, open_ms)."""
        h, body = self._one(cid, {"op": "score", "kind": kind, "sealed": True}, packet)
        if h.get("ok"):
            return True, (body, h.get("seal_ms", 0.0), bool(h.get("tuple"))), h.get("open_ms", 0.0)
        return False, body, h.get("open_ms", 0.0)

    def scores(self, kind, vec, cids):
        payload = vec_bytes(vec)
        got = self._collect({c: self._submit(c, {"op": "score", "kind": kind}, payload) for c in cids})
        out = {}
        for c in cids:
            h = got[c][0]
            out[c] = tuple(float(x) for x in h["score"]) if h["tuple"] else float(h["score"][0])
        return out

    # --- per-round log
    def _flush(self, before=None, wait_s=2.0):
        """Write mtls.jsonl lines for finished rounds (< before; all when None) once the drones' timing reports of
        their tasks have arrived (they come with each drone's next request, within milliseconds)."""
        t0 = time.time()
        while True:
            with self.lock:
                due = [r for r in self.awaiting.values() if r is not None and (before is None or r < before)]
            if not due or time.time() - t0 > wait_s:
                break
            time.sleep(0.005)
        with self.lock:
            rounds = sorted(r for r in self.acc if r >= 0 and r not in self.flushed and (before is None or r < before))
            lines = []
            for r in rounds:
                a = self.acc[r]
                lines.append({"round": r, "drones": [{"cid": c, **{k: (round(v, 3) if isinstance(v, float) else v)
                                                                   for k, v in a.get(c, {"bytes_up": 0, "bytes_down": 0,
                                                                   "rpc_ms_up": 0.0, "rpc_ms_down": 0.0,
                                                                   "tasks": 0}).items()}} for c in self.cids],
                              "server_wait_ms": round(self.wait_ms.get(r, 0.0), 3)})
                self.flushed.add(r)
        if self.out and lines:
            with open(os.path.join(self.out, "mtls.jsonl"), "a") as f:
                f.writelines(json.dumps(x) + "\n" for x in lines)

    def abort(self):
        """The job failed or was interrupted: stop the drones and the server without waiting for round logs."""
        self.close(error=True)

    def close(self, error=False):
        if self.closed:
            return
        self.closed = True
        if not error:
            self._flush()
        for c in self.cids:
            self.queues[c].put(({"op": "stop", "task": 0, "round": self.rnd}, b""))
        t0 = time.time()
        for p in self.procs.values():
            p.join(timeout=max(0.1, 10 - (time.time() - t0)))
        for p in self.procs.values():
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)
        self.server.stop(grace=1.0).wait(5)
        if self.meta.get("status") == "connected":
            self.meta["status"] = "closed"
        self.meta.update(finished=time.time(),
                         exit_codes={drone_cn(c): p.exitcode for c, p in self.procs.items()})
        self._write_meta()
        if self.own_live and not error:
            self.live["phase"] = "done"
            self._live_write()
