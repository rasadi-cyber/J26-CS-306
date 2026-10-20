"""Secure link (security.secure_link): byte-identical round trips, tampering and forged signatures rejected."""
import pytest
import torch
from cryptography.exceptions import InvalidSignature, InvalidTag

from fldos import securelink as sl


def test_round_trip_identical():
    ch = sl.SecureChannel(range(3), None)
    ch.begin_round(0)
    v = torch.randn(1000)
    v[0], v[1] = float("nan"), -0.0
    for c in range(3):
        out = ch.uplink(0, c, v, torch.zeros_like(v))
        assert sl.vec_bytes(out) == sl.vec_bytes(v)
        assert sl.vec_bytes(ch.vote_copy(0, c, v)) == sl.vec_bytes(v)
    ch.downlink(0, v)
    rec = ch.end_round()
    assert len(rec["uplink"]) == 3 and len(rec["downlink"]) == 3 and rec["vote"]["messages"] == 3
    assert rec["uplink"][0]["plain_hex"] != rec["uplink"][0]["cipher_hex"]


def test_tamper_rejected():
    d = sl.tamper_demo(torch.randn(100))
    assert d["error"] == "InvalidTag" and d["cipher_hex_before"] != d["cipher_hex_after"]
    assert d["resend"]["verify_ok"] and d["resend"]["bit_identical"]
    assert d["resend"]["nonce_hex"] != d["nonce_hex_before"]  # fresh nonce on the re-send


def test_tamper_nack_resend_accept():
    """Tampered uplink -> server rejects (InvalidTag) -> encrypted NACK to the drone -> same bytes re-sealed with a
    fresh nonce -> accepted bit-identical; the round carries on and the event is logged."""
    ch = sl.SecureChannel([0, 1], None, log=lambda m: None)
    ch.begin_round(0)
    v = torch.randn(50)
    ch.links[0].tamper_next = True
    out = ch.uplink(0, 0, v)
    assert sl.vec_bytes(out) == sl.vec_bytes(v)
    ch.uplink(0, 1, v)
    ch.links[1].tamper_next = True  # a drone asks for the global model again too
    ch.downlink(0, v)
    rec = ch.end_round()
    assert rec["retransmits"] == 2 and [e["dir"] for e in rec["events"]] == ["uplink", "downlink"]
    up = rec["events"][0]
    assert up["error"] == "InvalidTag" and up["recovered"] and up["byte"] == 7
    assert up["cipher_hex"] != up["cipher_hex_rx"] and up["resend_nonce_hex"] != up["nonce_hex"]
    assert up["nack_body"] == {"type": "nack", "for": "uplink", "round": 0, "attempt": 1, "error": "InvalidTag"}
    assert up["nack_bytes"] < 300 and up["resend_bytes"] == up["bytes"]
    assert rec["uplink"][0]["attempts"] == 2 and rec["uplink"][1]["attempts"] == 1
    assert rec["downlink"][1]["attempts"] == 2


def test_retries_exhausted_raise():
    ch = sl.SecureChannel([0], None, log=lambda m: None, tamper_prob=1.0, max_retries=2)
    ch.begin_round(0)
    with pytest.raises(RuntimeError, match="max_retries"):
        ch.uplink(0, 0, torch.randn(10))
    assert len(ch.rec["events"]) == 3


def test_ballot_round_trip_keeps_types():
    ch = sl.SecureChannel([0], None)
    ch.begin_round(0)
    assert ch.ballot(0, 0, 0.123456789) == 0.123456789
    s = ch.ballot(0, 0, (0.5, 0.01))
    assert s == (0.5, 0.01) and isinstance(s, tuple)


def test_tamper_rng_separate():
    """The interception draws come from their own seeded RNG: same seed -> same events, global RNGs untouched."""
    import random
    random.seed(5)
    torch.manual_seed(5)
    a, b = random.random(), torch.rand(1)
    evs = []
    for _ in range(2):
        random.seed(5)
        torch.manual_seed(5)
        ch = sl.SecureChannel(range(3), None, log=lambda m: None, tamper_prob=0.3, max_retries=10, seed=1)
        ch.begin_round(0)
        for c in range(3):
            ch.uplink(0, c, torch.ones(20))
        evs.append([(e["cid"], e["attempt"], e["byte"]) for e in ch.end_round()["events"]])
        assert random.random() == a and torch.equal(torch.rand(1), b)
    assert evs[0] == evs[1]


def test_forged_signature_rejected():
    ch = sl.SecureChannel([0, 1], None)
    plain = sl.vec_bytes(torch.randn(10))
    # drone 1 encrypts with drone 0's session key but signs with its own key: the server checks drone 0's key
    packet, _ = sl.seal(ch.keys[0]["uplink"], ch.drones[1].sign_key, {"cid": 0, "round": 0, "dir": "uplink"}, plain)
    with pytest.raises(InvalidSignature):
        sl.open_packet(ch.keys[0]["uplink"], ch.drone_pub[0], packet)
    with pytest.raises(InvalidTag):  # wrong session key
        sl.open_packet(ch.keys[1]["uplink"], ch.drone_pub[0], packet)
