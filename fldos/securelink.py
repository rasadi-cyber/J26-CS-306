"""Opt-in simulated secure link (security.secure_link: true; default off) between each drone and the aggregation
server, with real cryptography (`cryptography` package). Training results are bit-identical with it on or off: every
model that crosses a link is decrypted back to exactly the bytes that were sent (checked on every message).

Enrollment (once per job)
  operator CA = Ed25519 key. Each drone and the server get an Ed25519 signing key and an X25519 key exchange key;
  "certificate" = CA signature over {cid, sign_pub, kex_pub}; fingerprint = SHA-256 of body + signature (16 hex).
Session (per drone per job)
  X25519 ECDH (drone key x server key) -> HKDF-SHA256 (random per-job salt) -> one AES-256-GCM key per direction:
  uplink (drone -> server: its update), downlink (server -> drone: the global model), vote (server -> voter: the
  models a voter has to score), ballot (voter -> server: its scores), ctrl (re-send requests, either way).
Packet
  header json {cid, round, dir} (sent in clear, authenticated as AES-GCM AAD) + 12-byte random nonce +
  ciphertext + 16-byte tag + Ed25519 signature by the sender over header + plaintext (binds the update to its
  round and direction). Plaintext = the model vector as float32 little-endian bytes (ballot: float64 scores).
Receiver
  tag check + decrypt (InvalidTag on any changed byte) -> signature check against the sender's enrolled key ->
  bytes -> tensor; asserted byte-identical to what was sent.
Retransmission
  A packet that fails the receiver's checks (InvalidTag, InvalidSignature, or a packet that cannot be parsed) is
  dropped and the receiver sends a NACK (re-send request: a small signed + encrypted ctrl message on the same link,
  the other way). The sender re-seals the SAME bytes with a fresh nonce and sends again, up to security.max_retries
  (default 3) re-sends; if every attempt fails the job stops with an error (an update is never silently dropped).
  Logged per round: `retransmits` and one `events` record per rejected packet.
Interception simulation (opt-in, dashboard demos)
  security.tamper_prob = p (default 0): every data packet (not the NACKs) independently gets one ciphertext byte
  flipped on the wire with probability p. The draws come from their own RNG (random.Random(seed + 9001)), so the
  training RNGs, and with them the results, are unchanged. NACKs are not tampered: a lost NACK would end in the same
  re-send after a timeout.
Link
  in-memory queues with simulated transfer time = latency + bytes * 8 / bandwidth (default 10 Mbit/s, 20 ms).
  Link.tamper_next flips one ciphertext byte of the next packet on the wire (set by the interception simulation and
  by tamper_demo).
"""
import hashlib
import json
import os
import random
import struct
import time
from collections import deque

import numpy as np
import torch
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ALGOS = {"kex": "X25519", "aead": "AES-256-GCM", "sig": "Ed25519", "kdf": "HKDF-SHA256"}
DIRS = ("uplink", "downlink", "vote", "ballot", "ctrl")
UP_DIRS = ("uplink", "ballot")  # sent by the drone; downlink and vote are sent by the server
SERVER, DRONE = "server", "drone"
N_HEX = 48     # bytes of plaintext / ciphertext logged per message
N_SAMPLE = 8   # floats logged per vector
TAMPER_SEED = 9001  # interception draws: random.Random(seed + TAMPER_SEED), separate from every training RNG


def enabled(cfg):
    return bool(cfg.get("security", {}).get("secure_link"))


def _raw(pub):
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _now_ms(t0):
    return (time.perf_counter() - t0) * 1000.0


def vec_bytes(vec):
    """Model vector -> float32 little-endian bytes (the plaintext)."""
    a = vec.detach().cpu().numpy()
    assert a.dtype == np.float32, a.dtype
    return a.astype("<f4", copy=False).tobytes()


def bytes_vec(b, like=None):
    """float32 little-endian bytes -> tensor (own, writable copy; on like's device)."""
    t = torch.from_numpy(np.frombuffer(b, dtype="<f4").astype(np.float32, copy=True))
    return t.to(like.device) if like is not None else t


def score_bytes(score):
    """A voter's score (a float or a tuple of floats) -> float64 little-endian bytes (ballot plaintext)."""
    xs = score if isinstance(score, (tuple, list)) else (score,)
    return struct.pack(f"<{len(xs)}d", *xs)


def bytes_score(b, like):
    """Ballot bytes -> score of the same shape and number types as `like`."""
    xs = struct.unpack(f"<{len(b) // 8}d", b)
    if isinstance(like, (tuple, list)):
        return type(like)(type(o)(x) for o, x in zip(like, xs))
    return type(like)(xs[0])


def sample(vec):
    return [float(x) for x in vec.detach().cpu()[:N_SAMPLE].tolist()]


# ---------------------------------------------------------------- enrollment
class Party:
    """One enrolled identity (a drone or the server): Ed25519 signing key + X25519 key exchange key."""

    def __init__(self, ident):
        self.id = ident
        self.sign_key = Ed25519PrivateKey.generate()
        self.kex_key = X25519PrivateKey.generate()
        self.sign_pub = _raw(self.sign_key.public_key())
        self.kex_pub = _raw(self.kex_key.public_key())
        self.cert = None


class Authority:
    """The operator's certificate authority (Ed25519)."""

    def __init__(self):
        self.key = Ed25519PrivateKey.generate()
        self.pub = _raw(self.key.public_key())
        self.fingerprint = hashlib.sha256(self.pub).hexdigest()[:16]

    def certify(self, p):
        body = json.dumps({"cid": p.id, "sign_pub": p.sign_pub.hex(), "kex_pub": p.kex_pub.hex()},
                          sort_keys=True, separators=(",", ":")).encode()
        sig = self.key.sign(body)
        p.cert = {"body": body, "sig": sig, "fingerprint": hashlib.sha256(body + sig).hexdigest()[:16]}
        return p.cert

    def check(self, cert):
        """Verify a certificate; return its (sign_pub, kex_pub) public keys (raises InvalidSignature)."""
        Ed25519PublicKey.from_public_bytes(self.pub).verify(cert["sig"], cert["body"])
        b = json.loads(cert["body"])
        return (Ed25519PublicKey.from_public_bytes(bytes.fromhex(b["sign_pub"])),
                X25519PublicKey.from_public_bytes(bytes.fromhex(b["kex_pub"])))


def session_keys(own_kex, peer_kex_pub, cid, salt):
    """X25519 ECDH -> HKDF-SHA256 -> one AES-256 key per direction (both ends derive the same keys)."""
    shared = own_kex.exchange(peer_kex_pub)
    okm = HKDF(algorithm=hashes.SHA256(), length=32 * len(DIRS), salt=salt,
               info=f"fldos-securelink v1 cid={cid}".encode()).derive(shared)
    return {d: okm[32 * i:32 * (i + 1)] for i, d in enumerate(DIRS)}


# ---------------------------------------------------------------- packets
def seal(key, sign_key, header, plain):
    """-> (packet bytes, parts). Sign header + plaintext, then AES-256-GCM with the header as AAD."""
    aad = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    sig = sign_key.sign(aad + plain)
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plain, aad)  # ciphertext || 16-byte tag
    packet = struct.pack("<H", len(aad)) + aad + nonce + struct.pack("<I", len(ct)) + ct + sig
    return packet, {"nonce": nonce, "ct": ct, "sig": sig}


def parse(packet):
    """packet -> (aad, header, nonce, ct, sig, ciphertext offset)."""
    n = struct.unpack_from("<H", packet, 0)[0]
    aad = packet[2:2 + n]
    nonce = packet[2 + n:14 + n]
    m = struct.unpack_from("<I", packet, 14 + n)[0]
    off = 18 + n
    return aad, json.loads(aad), nonce, packet[off:off + m], packet[off + m:], off


def open_packet(key, verify_pub, packet):
    """Tag check + decrypt (InvalidTag), then the sender's signature (InvalidSignature). -> (header, plaintext)."""
    aad, header, nonce, ct, sig, _ = parse(packet)
    plain = AESGCM(key).decrypt(nonce, ct, aad)
    verify_pub.verify(sig, aad + plain)
    return header, plain


class Link:
    """Simulated network link between one drone and the server: two in-memory queues + counters."""

    def __init__(self, cid, bandwidth_mbps=10.0, latency_ms=20.0):
        self.cid, self.bandwidth_mbps, self.latency_ms = cid, float(bandwidth_mbps), float(latency_ms)
        self.server_inbox, self.drone_inbox = deque(), deque()
        self.tamper_next = False  # flip one ciphertext byte of the next packet on the wire
        self.tamper_pos = 7       # which ciphertext byte (inside the logged first 48 bytes)
        self.packets = {d: 0 for d in DIRS}
        self.bytes = {d: 0 for d in DIRS}
        self.sim_s = 0.0
        self.tampered = []

    def transfer_s(self, n_bytes):
        return self.latency_ms / 1000.0 + n_bytes * 8 / (self.bandwidth_mbps * 1e6)

    def send(self, packet, to=None):
        """Put a packet on the wire (uplink / ballot -> server inbox, downlink / vote -> drone inbox; ctrl: `to`);
        -> simulated seconds."""
        _, header, _, ct, _, off = parse(packet)
        if self.tamper_next:
            self.tamper_next = False
            i = off + min(self.tamper_pos, len(ct) - 1)
            packet = packet[:i] + bytes([packet[i] ^ 0x01]) + packet[i + 1:]
            self.tampered.append({"round": header["round"], "dir": header["dir"], "byte": i - off})
        d = header["dir"]
        to = to or (SERVER if d in UP_DIRS else DRONE)
        (self.server_inbox if to == SERVER else self.drone_inbox).append(packet)
        self.packets[d] += 1
        self.bytes[d] += len(packet)
        t = self.transfer_s(len(packet))
        self.sim_s += t
        return t

    def receive(self, at=SERVER):
        return (self.server_inbox if at == SERVER else self.drone_inbox).popleft()


# ---------------------------------------------------------------- channel used by runner.run_job
class SecureChannel:
    """Enrolls the swarm, keeps one Link per drone, moves every model through it and logs each round.

    out_dir: job folder -> securelink.jsonl (one line per round), securelink_meta.json, live_state.json
    (dashboard live view, atomically overwritten on every state change). out_dir None: nothing is written.
    tamper_prob / max_retries / seed: interception simulation and retransmission (see the module docstring)."""

    def __init__(self, cids, out_dir=None, bandwidth_mbps=10.0, latency_ms=20.0, log=print, tamper_prob=0.0,
                 max_retries=3, seed=0):
        self.cids, self.out, self.log = list(cids), out_dir, log
        self.tamper_prob, self.max_retries = float(tamper_prob or 0.0), int(max_retries)
        self.tamper_rng = random.Random(int(seed) + TAMPER_SEED)  # never shared with training
        t0 = time.perf_counter()
        self.ca = Authority()
        self.server = Party(SERVER)
        self.drones = {c: Party(c) for c in self.cids}
        for p in [self.server, *self.drones.values()]:
            self.ca.certify(p)
        self.salt = os.urandom(16)
        # Each side checks the other's certificate before deriving keys.
        srv_sign, srv_kex = self.ca.check(self.server.cert)
        self.drone_pub, self.keys = {}, {}
        for c, p in self.drones.items():
            d_sign, d_kex = self.ca.check(p.cert)          # server checks the drone
            k_server = session_keys(self.server.kex_key, d_kex, c, self.salt)
            k_drone = session_keys(p.kex_key, srv_kex, c, self.salt)
            assert k_server == k_drone
            self.drone_pub[c], self.keys[c] = d_sign, k_server
        self.server_pub = srv_sign
        self.enroll_ms = _now_ms(t0)
        self.links = {c: Link(c, bandwidth_mbps, latency_ms) for c in self.cids}
        self.totals = {"messages": 0, "bytes": 0, "crypto_ms": 0.0, "rejected": 0, "retransmits": 0}
        self.rnd, self.rec, self.live = None, None, None
        if self.out:
            open(os.path.join(self.out, "securelink.jsonl"), "w").close()
            self._write_meta()

    # --- metadata
    def meta(self):
        l0 = self.links[self.cids[0]]
        return {**ALGOS, "bandwidth_mbps": l0.bandwidth_mbps, "latency_ms": l0.latency_ms,
                "tamper_prob": self.tamper_prob, "max_retries": self.max_retries,
                "ca_fingerprint": self.ca.fingerprint, "server_fingerprint": self.server.cert["fingerprint"],
                "enrollment": [{"cid": c, "fingerprint": p.cert["fingerprint"], "sign_pub": p.sign_pub.hex(),
                                "kex_pub": p.kex_pub.hex()} for c, p in self.drones.items()],
                "hkdf_salt": self.salt.hex(), "enroll_ms": round(self.enroll_ms, 3), "totals": dict(self.totals)}

    def _write_meta(self):
        _write_json(os.path.join(self.out, "securelink_meta.json"), self.meta())

    # --- live state (dashboard)
    def _live_write(self):
        if self.out and self.live is not None:
            _write_json(os.path.join(self.out, "live_state.json"), self.live, indent=None)

    def phase(self, name):
        if self.live is not None:
            self.live["phase"] = name
            self._live_write()

    def _mark(self, cid, state, **extra):
        if self.live is not None:
            d = self.live["drones"][self.cids.index(cid)]
            for k in ("dir", "attempt"):
                d.pop(k, None)
            d.update(state=state, t=time.time(), **extra)
            self._live_write()

    # --- round bookkeeping
    def begin_round(self, rnd):
        self.rnd = rnd
        self.rec = {"round": rnd, "uplink": [], "downlink": [],
                    "vote": {"messages": 0, "bytes": 0, "crypto_ms": 0.0, "ballots": 0, "ballot_bytes": 0},
                    "retransmits": 0, "events": [], "totals": {"bytes": 0, "crypto_ms": 0.0}}
        self.local_ms = {}
        now = time.time()
        self.live = {"round": rnd, "phase": "local_training", "retransmits": 0,
                     "drones": [{"cid": c, "state": "training", "t": now} for c in self.cids]}
        self._live_write()

    def trained(self, cid, ms):
        """A drone's local training finished (ms = its training time)."""
        self.local_ms[cid] = round(float(ms), 3)
        self._mark(cid, "ready")

    def _count(self, n_bytes, ms, n=1):
        if self.rec is not None:  # None: the starting model handed out before round 1 (job totals only)
            self.rec["totals"]["bytes"] += n_bytes
            self.rec["totals"]["crypto_ms"] += ms
        self.totals["messages"] += n
        self.totals["bytes"] += n_bytes
        self.totals["crypto_ms"] += ms

    def _intercepted(self):
        """Interception simulation: does this data packet get a byte flipped on the wire? (own RNG)"""
        return self.tamper_prob > 0 and self.tamper_rng.random() < self.tamper_prob

    def _nack(self, cid, direction, attempt, error):
        """The receiver of a rejected packet asks its sender for a re-send: a small signed, encrypted ctrl message on
        the same link, going the other way. -> (packet, parts, simulated seconds, crypto ms, body)."""
        from_drone = direction in UP_DIRS  # the rejected packet came from the drone -> the server sends the NACK
        sender, verify, to = ((self.server.sign_key, self.server_pub, DRONE) if from_drone
                              else (self.drones[cid].sign_key, self.drone_pub[cid], SERVER))
        body = {"type": "nack", "for": direction, "round": self.rnd, "attempt": attempt, "error": error}
        key, link = self.keys[cid]["ctrl"], self.links[cid]
        t0 = time.perf_counter()
        packet, parts = seal(key, sender, {"cid": cid, "round": self.rnd, "dir": "ctrl"},
                             json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
        t = link.send(packet, to=to)
        _, msg = open_packet(key, verify, link.receive(to))
        assert json.loads(msg) == body
        return packet, parts, t, _now_ms(t0), body

    def _transfer(self, cid, direction, plain):
        """Seal at the sender, send over the drone's link, open at the receiver. A packet that fails the checks is
        NACKed and the same bytes are re-sealed (fresh nonce) and re-sent, up to max_retries times.
        -> (decrypted bytes, parts of the accepted packet, stats)."""
        link, key = self.links[cid], self.keys[cid][direction]
        from_drone = direction in UP_DIRS
        sender, verify, receiver = ((self.drones[cid].sign_key, self.drone_pub[cid], SERVER) if from_drone
                                    else (self.server.sign_key, self.server_pub, DRONE))
        header = {"cid": cid, "round": self.rnd, "dir": direction}
        enc_ms = dec_ms = transfer_s = 0.0
        n_bytes = n_msgs = 0
        ev = None
        for attempt in range(1, self.max_retries + 2):
            t0 = time.perf_counter()
            packet, parts = seal(key, sender, header, plain)
            enc_ms += _now_ms(t0)
            if ev is not None:  # this packet is the re-send the last NACK asked for
                ev.update(resend_bytes=len(packet), resend_nonce_hex=parts["nonce"].hex(),
                          resend_cipher_hex=parts["ct"][:N_HEX].hex())
            if self._intercepted():
                link.tamper_next, link.tamper_pos = True, self.tamper_rng.randrange(N_HEX)
            n_tampered = len(link.tampered)
            transfer_s += link.send(packet)
            n_bytes, n_msgs = n_bytes + len(packet), n_msgs + 1
            got = link.receive(receiver)
            t0 = time.perf_counter()
            try:
                _, dec = open_packet(key, verify, got)
                dec_ms += _now_ms(t0)
                if ev is not None:
                    ev["recovered"] = True
                break
            except (InvalidTag, InvalidSignature, ValueError, KeyError, struct.error) as e:
                dec_ms += _now_ms(t0)
                error = type(e).__name__ if isinstance(e, (InvalidTag, InvalidSignature)) else "ParseError"
            self.totals["rejected"] += 1
            try:
                ct_rx = parse(got)[3][:N_HEX].hex()
            except Exception:
                ct_rx = None
            ev = {"cid": cid, "dir": direction, "attempt": attempt, "error": error,
                  "byte": link.tampered[-1]["byte"] if len(link.tampered) > n_tampered else None,
                  "bytes": len(packet), "nonce_hex": parts["nonce"].hex(), "tag_hex": parts["ct"][-16:].hex(),
                  "cipher_hex": parts["ct"][:N_HEX].hex(), "cipher_hex_rx": ct_rx, "recovered": False}
            self.rec["events"].append(ev)
            if attempt > self.max_retries:
                self.log(f"[securelink] round {self.rnd} drone {cid} {direction}: packet rejected {attempt} times "
                         f"(last: {error}); stopping")
                self._count(n_bytes, enc_ms + dec_ms, n_msgs)
                raise RuntimeError(f"secure link: {direction} packet of drone {cid} rejected {attempt} times in round "
                                   f"{self.rnd} (last: {error}); giving up after security.max_retries = "
                                   f"{self.max_retries} re-sends")
            npk, nparts, nt, nms, body = self._nack(cid, direction, attempt, error)
            ev.update(nack_bytes=len(npk), nack_nonce_hex=nparts["nonce"].hex(), nack_hex=nparts["ct"][:N_HEX].hex(),
                      nack_body=body, nack_transfer_s=round(nt, 6))
            transfer_s += nt
            n_bytes, n_msgs, enc_ms = n_bytes + len(npk), n_msgs + 1, enc_ms + nms
            self.rec["retransmits"] += 1
            self.totals["retransmits"] += 1
            if self.live is not None:
                self.live["retransmits"] = self.rec["retransmits"]
                self.live["event"] = {"cid": cid, "dir": direction, "attempt": attempt + 1, "error": error}
            self._mark(cid, "resending", dir=direction, attempt=attempt + 1)
        assert dec == plain, "decrypted message differs from the sent message"
        self._count(n_bytes, enc_ms + dec_ms, n_msgs)
        return dec, parts, {"bytes": len(packet), "enc_ms": round(enc_ms, 3), "dec_ms": round(dec_ms, 3),
                            "transfer_s": round(transfer_s, 6), "attempts": attempt}

    # --- real drone processes (security.transport = "mtls"): each drone seals and opens its own messages
    def drone_material(self, cid):
        """Keys provisioned to drone cid at enrollment (operator's hangar): its Ed25519 signing key, its session
        keys per direction and the server's public signing key. Only this drone's process receives them."""
        p = self.drones[cid]
        return {"sign": p.sign_key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                                 serialization.NoEncryption()).hex(),
                "keys": {d: k.hex() for d, k in self.keys[cid].items()}, "server_sign_pub": _raw(self.server_pub).hex(),
                "cid": cid}

    def _wire(self, cid, packet, to):
        """Put a packet on drone cid's link (interception simulation decides here) and take it off at `to`."""
        link = self.links[cid]
        if self._intercepted():
            link.tamper_next, link.tamper_pos = True, self.tamper_rng.randrange(N_HEX)
        n = len(link.tampered)
        t = link.send(packet, to=to)
        return link.receive(to), t, (link.tampered[-1]["byte"] if len(link.tampered) > n else None)

    def _server_nack(self, cid, direction, attempt, error):
        """Server -> drone re-send request (signed + encrypted with the ctrl key) -> (packet, ms, body)."""
        body = {"type": "nack", "for": direction, "round": self.rnd, "attempt": attempt, "error": error}
        t0 = time.perf_counter()
        packet, _ = seal(self.keys[cid]["ctrl"], self.server.sign_key, {"cid": cid, "round": self.rnd, "dir": "ctrl"},
                         json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
        return packet, _now_ms(t0), body

    def _event(self, cid, direction, attempt, error, packet, got, byte):
        _, _, nonce, ct, _, _ = parse(packet)
        try:
            ct_rx = parse(got)[3][:N_HEX].hex()
        except Exception:
            ct_rx = None
        ev = {"cid": cid, "dir": direction, "attempt": attempt, "error": error, "byte": byte, "bytes": len(packet),
              "nonce_hex": nonce.hex(), "tag_hex": ct[-16:].hex(), "cipher_hex": ct[:N_HEX].hex(),
              "cipher_hex_rx": ct_rx, "recovered": False}
        self.rec["events"].append(ev)
        self.totals["rejected"] += 1
        return ev

    def _retry_note(self, cid, direction, attempt, error, ev, nack, nack_ms, body):
        ev.update(nack_bytes=len(nack), nack_nonce_hex=parse(nack)[2].hex(), nack_hex=parse(nack)[3][:N_HEX].hex(),
                  nack_body=body)
        self.rec["retransmits"] += 1
        self.totals["retransmits"] += 1
        if self.live is not None:
            self.live["retransmits"] = self.rec["retransmits"]
            self.live["event"] = {"cid": cid, "dir": direction, "attempt": attempt + 1, "error": error}
        self._mark(cid, "resending", dir=direction, attempt=attempt + 1)

    def _too_many(self, cid, direction, attempt, error):
        if attempt > self.max_retries:
            raise RuntimeError(f"secure link: {direction} packet of drone {cid} rejected {attempt} times in round "
                               f"{self.rnd} (last: {error}); giving up after security.max_retries = "
                               f"{self.max_retries} re-sends")

    def receive_from_drone(self, cid, direction, packet, seal_ms, reseal):
        """A packet sealed by the drone process (uplink / ballot): wire -> tag check, decrypt, signature check at
        the server. A rejected packet -> server NACK -> the drone process re-seals (fresh nonce) via reseal(cid,
        direction, nack) -> (packet, seal_ms). -> (plaintext, accepted packet, stats)."""
        enc_ms, dec_ms, transfer_s, n_bytes, n_msgs, ev = seal_ms, 0.0, 0.0, 0, 0, None
        for attempt in range(1, self.max_retries + 2):
            if ev is not None:
                _, _, nonce, ct, _, _ = parse(packet)
                ev.update(resend_bytes=len(packet), resend_nonce_hex=nonce.hex(), resend_cipher_hex=ct[:N_HEX].hex())
            got, t, byte = self._wire(cid, packet, SERVER)
            transfer_s, n_bytes, n_msgs = transfer_s + t, n_bytes + len(packet), n_msgs + 1
            t0 = time.perf_counter()
            try:
                _, plain = open_packet(self.keys[cid][direction], self.drone_pub[cid], got)
                dec_ms += _now_ms(t0)
                if ev is not None:
                    ev["recovered"] = True
                break
            except (InvalidTag, InvalidSignature, ValueError, KeyError, struct.error) as e:
                dec_ms += _now_ms(t0)
                error = type(e).__name__ if isinstance(e, (InvalidTag, InvalidSignature)) else "ParseError"
            ev = self._event(cid, direction, attempt, error, packet, got, byte)
            self._too_many(cid, direction, attempt, error)
            nack, nms, body = self._server_nack(cid, direction, attempt, error)
            transfer_s += self.links[cid].transfer_s(len(nack))
            self.links[cid].bytes["ctrl"] += len(nack)
            self.links[cid].packets["ctrl"] += 1
            ev["nack_transfer_s"] = round(self.links[cid].transfer_s(len(nack)), 6)
            self._retry_note(cid, direction, attempt, error, ev, nack, nms, body)
            packet, sms = reseal(cid, direction, nack)
            enc_ms, n_bytes, n_msgs = enc_ms + nms + sms, n_bytes + len(nack), n_msgs + 1
        self._count(n_bytes, enc_ms + dec_ms, n_msgs)
        return plain, packet, {"bytes": len(packet), "enc_ms": round(enc_ms, 3), "dec_ms": round(dec_ms, 3),
                               "transfer_s": round(transfer_s, 6), "attempts": attempt}

    def send_to_drone(self, cid, direction, plain, deliver):
        """Server -> drone process (downlink / vote): seal at the server, wire, deliver(cid, packet) -> the drone
        opens and verifies it and answers (ok, reply, open_ms) or (False, drone NACK packet, open_ms); a rejected
        packet is re-sealed (fresh nonce) and re-sent. -> (drone's reply, stats)."""
        enc_ms, dec_ms, transfer_s, n_bytes, n_msgs, ev = 0.0, 0.0, 0.0, 0, 0, None
        header = {"cid": cid, "round": self.rnd, "dir": direction}
        for attempt in range(1, self.max_retries + 2):
            t0 = time.perf_counter()
            packet, parts = seal(self.keys[cid][direction], self.server.sign_key, header, plain)
            enc_ms += _now_ms(t0)
            if ev is not None:
                ev.update(resend_bytes=len(packet), resend_nonce_hex=parts["nonce"].hex(),
                          resend_cipher_hex=parts["ct"][:N_HEX].hex())
            got, t, byte = self._wire(cid, packet, DRONE)
            transfer_s, n_bytes, n_msgs = transfer_s + t, n_bytes + len(packet), n_msgs + 1
            ok, reply, open_ms = deliver(cid, got)
            dec_ms += open_ms
            if ok:
                if ev is not None:
                    ev["recovered"] = True
                break
            nack = reply  # the drone's re-send request, signed + encrypted by the drone process
            _, body = open_packet(self.keys[cid]["ctrl"], self.drone_pub[cid], nack)
            body = json.loads(body)
            error = body.get("error", "InvalidTag")
            ev = self._event(cid, direction, attempt, error, packet, got, byte)
            self._too_many(cid, direction, attempt, error)
            transfer_s += self.links[cid].transfer_s(len(nack))
            self.links[cid].bytes["ctrl"] += len(nack)
            self.links[cid].packets["ctrl"] += 1
            ev["nack_transfer_s"] = round(self.links[cid].transfer_s(len(nack)), 6)
            self._retry_note(cid, direction, attempt, error, ev, nack, 0.0, body)
            n_bytes, n_msgs = n_bytes + len(nack), n_msgs + 1
        self._count(n_bytes, enc_ms + dec_ms, n_msgs)
        return reply, {"bytes": len(packet), "enc_ms": round(enc_ms, 3), "dec_ms": round(dec_ms, 3),
                       "transfer_s": round(transfer_s, 6), "attempts": attempt}

    def uplink_remote(self, rnd, cid, packet, seal_ms, reseal, like, ref=None, local_ms=None):
        """Drone process cid's sealed update -> the server's decrypted copy (tensor like `like`)."""
        self.rnd = rnd
        plain, acc, st = self.receive_from_drone(cid, "uplink", packet, seal_ms, reseal)
        out = bytes_vec(plain, like)
        _, _, nonce, ct, sig, _ = parse(acc)
        self.rec["uplink"].append({
            "cid": cid, **st, "verify_ok": True, "local_train_ms": local_ms if local_ms is not None else self.local_ms.get(cid),
            "update_norm": round(float((out - ref).norm()), 6) if ref is not None else None,
            "nonce_hex": nonce.hex(), "tag_hex": ct[-16:].hex(), "sig_hex": sig.hex(),
            "plain_hex": plain[:N_HEX].hex(), "cipher_hex": ct[:N_HEX].hex(),
            "plain_sample": sample(out), "decrypted_sample": sample(out), "sealed_by": "drone process"})
        self._mark(cid, "sent")
        return out

    def downlink_remote(self, rnd, global_vec, deliver, record=True):
        """New global model -> every drone process (each opens it with its own keys). record False: the initial
        model handed out before round 1 (no interception draw, not part of a round's log)."""
        self.rnd = rnd
        if record:
            self.phase("downlink")
        plain = vec_bytes(global_vec)
        for cid in self.cids:
            if record:
                _, st = self.send_to_drone(cid, "downlink", plain, deliver)
                self.rec["downlink"].append({"cid": cid, **st, "opened_by": "drone process"})
                self._mark(cid, "received")
            else:
                tp, _ = self.tamper_prob, None
                self.tamper_prob = 0.0
                try:
                    self.send_to_drone(cid, "downlink", plain, deliver)
                finally:
                    self.tamper_prob = tp
        if record:
            self.rec["global_sample"] = sample(global_vec)
        return global_vec

    def remote_voter_scores(self, pool):
        """Votes with real drone processes: the model to score goes to each voter sealed (vote); the voter opens it,
        scores it on its own data and returns its ballot sealed by itself; the server opens the ballots."""
        def voter_scores(vec, cids, kind="voter_score"):
            cids = list(cids)
            if kind not in ("voter_score", "voter_score_se"):
                return pool.scores(kind, vec, cids)
            plain, out, v = vec_bytes(vec), {}, self.rec["vote"]
            for c in cids:
                reply, st = self.send_to_drone(c, "vote", plain, lambda cid, pk: pool.score_sealed(kind, cid, pk))
                v["messages"] += 1
                v["bytes"] += st["bytes"]
                v["crypto_ms"] += st["enc_ms"] + st["dec_ms"]
                ballot, seal_ms, is_tuple = reply
                bplain, _, bst = self.receive_from_drone(c, "ballot", ballot, seal_ms, pool.reseal)
                v["ballots"] += 1
                v["ballot_bytes"] += bst["bytes"]
                v["crypto_ms"] += bst["enc_ms"] + bst["dec_ms"]
                xs = struct.unpack(f"<{len(bplain) // 8}d", bplain)
                out[c] = tuple(float(x) for x in xs) if is_tuple else float(xs[0])
                self._restore(c, {"attempts": max(st["attempts"], bst["attempts"])})
            return out
        return voter_scores

    def uplink(self, rnd, cid, vec, ref=None):
        """Drone cid's update -> server. Returns the server's decrypted copy (byte-identical to vec).
        ref: the global model the drone started from (only for the logged update norm)."""
        self.rnd = rnd
        plain = vec_bytes(vec)
        dec, parts, st = self._transfer(cid, "uplink", plain)
        out = bytes_vec(dec, vec)
        ct = parts["ct"]
        self.rec["uplink"].append({
            "cid": cid, **st, "verify_ok": True, "local_train_ms": self.local_ms.get(cid),
            "update_norm": round(float((vec - ref).norm()), 6) if ref is not None else None,
            "nonce_hex": parts["nonce"].hex(), "tag_hex": ct[-16:].hex(), "sig_hex": parts["sig"].hex(),
            "plain_hex": plain[:N_HEX].hex(), "cipher_hex": ct[:N_HEX].hex(),
            "plain_sample": sample(vec), "decrypted_sample": sample(out)})
        self._mark(cid, "sent")
        return out

    def downlink(self, rnd, global_vec):
        """New global model -> every drone (each on its own link and key). Returns global_vec."""
        self.rnd = rnd
        self.phase("downlink")
        plain = vec_bytes(global_vec)
        for cid in self.cids:
            _, _, st = self._transfer(cid, "downlink", plain)
            self.rec["downlink"].append({"cid": cid, **st})
            self._mark(cid, "received")
        self.rec["global_sample"] = sample(global_vec)
        return global_vec

    def vote_copy(self, rnd, voter_cid, vec):
        """A model sent by the server to one voter to be scored (direction vote). Returns the voter's decrypted copy."""
        self.rnd = rnd
        dec, _, st = self._transfer(voter_cid, "vote", vec_bytes(vec))
        v = self.rec["vote"]
        v["messages"] += 1
        v["bytes"] += st["bytes"]
        v["crypto_ms"] += st["enc_ms"] + st["dec_ms"]
        self._restore(voter_cid, st)
        return bytes_vec(dec, vec)

    def ballot(self, rnd, voter_cid, score):
        """A voter's score of one model -> server (direction ballot). Returns the server's decrypted score (same
        value and number type)."""
        self.rnd = rnd
        dec, _, st = self._transfer(voter_cid, "ballot", score_bytes(score))
        v = self.rec["vote"]
        v["ballots"] += 1
        v["ballot_bytes"] += st["bytes"]
        v["crypto_ms"] += st["enc_ms"] + st["dec_ms"]
        self._restore(voter_cid, st)
        return bytes_score(dec, score)

    def _restore(self, cid, st):
        """After a re-sent vote copy / ballot got through, the drone is back to its state before the vote."""
        if st["attempts"] > 1:
            self._mark(cid, "received" if self.rec["downlink"] else "sent")

    def wrap_voter_scores(self, fn):
        """Wrap FedSim.voter_scores for the vote: every model the voters score (the filtered model M_B, kind
        voter_score_se, and each held-back model, kind voter_score) is first sent to each voter over its vote link,
        and each voter's score comes back over its ballot link. Both are checked byte-identical, so the votes are
        computed exactly as before (same call, same order, serial or in the drone pool)."""
        def voter_scores(vec, cids, kind="voter_score"):
            cids = list(cids)
            if kind not in ("voter_score", "voter_score_se"):
                return fn(vec, cids, kind)
            for c in cids:
                self.vote_copy(self.rnd, c, vec)
            scores = fn(vec, cids, kind)
            return {c: self.ballot(self.rnd, c, s) for c, s in scores.items()}
        return voter_scores

    def end_round(self):
        r = self.rec
        r["totals"]["crypto_ms"] = round(r["totals"]["crypto_ms"], 3)
        r["totals"]["retransmits"] = r["retransmits"]
        r["vote"]["crypto_ms"] = round(r["vote"]["crypto_ms"], 3)
        if r["retransmits"]:  # one log line per round (each event is in securelink.jsonl)
            ev = r["events"]
            self.log(f"[securelink] round {r['round']}: {len(ev)} packet(s) rejected "
                     f"({', '.join(sorted(set(e['error'] for e in ev)))}) and re-sent after a NACK: "
                     + ", ".join(f"drone {e['cid']} {e['dir']}" for e in ev[:6]) + (" ..." if len(ev) > 6 else ""))
        if self.out:
            with open(os.path.join(self.out, "securelink.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
        return r

    def close(self):
        self.totals["crypto_ms"] = round(self.totals["crypto_ms"], 3)
        if self.out:
            self._write_meta()
            if self.live is not None:
                self.live["phase"] = "done"
                self._live_write()


def _write_json(path, obj, indent=2):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def event_story(ev, accepted):
    """One logged interception event + the uplink record that finally got through -> the demo's sequence:
    original packet, flipped byte, rejection, NACK, re-sent packet, accepted (bit-identical)."""
    same = bool(accepted.get("plain_sample")) and accepted.get("plain_sample") == accepted.get("decrypted_sample")
    return {"round": ev.get("round"), "cid": ev["cid"], "dir": ev["dir"], "attempt": ev["attempt"],
            "error": ev["error"], "flipped_byte": ev.get("byte"), "bytes": ev["bytes"],
            "nonce_hex_before": ev["nonce_hex"], "tag_hex": ev["tag_hex"],
            "cipher_hex_before": ev["cipher_hex"], "cipher_hex_after": ev["cipher_hex_rx"], "accepted": False,
            "nack": {"bytes": ev.get("nack_bytes"), "nonce_hex": ev.get("nack_nonce_hex"), "hex": ev.get("nack_hex"),
                     "body": ev.get("nack_body"), "transfer_s": ev.get("nack_transfer_s")},
            "resend": {"bytes": ev.get("resend_bytes"), "nonce_hex": ev.get("resend_nonce_hex"),
                       "cipher_hex": ev.get("resend_cipher_hex"), "verify_ok": bool(ev.get("recovered")),
                       "final_nonce_hex": accepted.get("nonce_hex"), "final_cipher_hex": accepted.get("cipher_hex"),
                       "attempts": accepted.get("attempts"), "plain_sample": accepted.get("plain_sample"),
                       "decrypted_sample": accepted.get("decrypted_sample"), "bit_identical": same}}


def tamper_demo(vec, cid=0, rnd=0):
    """Demo on a separate, throwaway enrollment and link (never a training run's): one uplink packet has one
    ciphertext byte flipped on the wire; the server's tag check rejects it (InvalidTag), NACKs it, and the drone's
    re-sent packet (same bytes, fresh nonce) is accepted bit-identical."""
    ch = SecureChannel([cid], out_dir=None, log=lambda m: None)
    ch.begin_round(rnd)
    ch.links[cid].tamper_next = True
    ch.uplink(rnd, cid, vec)
    rec = ch.end_round()
    ev = dict(rec["events"][0], round=rnd)
    assert ev["error"] == "InvalidTag" and ev["recovered"], "tampered packet was accepted or not recovered"
    return {**event_story(ev, rec["uplink"][0]), "source": "throwaway enrollment"}
