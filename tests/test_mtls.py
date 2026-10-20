"""Real mTLS transport (security.transport = "mtls"): drone processes over gRPC + TLS 1.3 give bit-identical results to
the in-process path; a revoked drone certificate is refused (UNAUTHENTICATED). Tiny synthetic partition, 3 drones."""
import json
import os

import numpy as np
import pytest
import torch

from fldos import mtls
from fldos.data import Client, FedData, save_shared
from fldos.sim import FedSim

CFG = {"data": {"classes": ["Normal", "A", "B"], "normal_class": "Normal", "dos_classes": ["A", "B"]},
       "training": {"device": "cpu", "hidden": [16], "dropout": 0.1, "class_weighting": "inverse", "local_steps": 5,
                    "batch_size": 32, "optimizer": "sgd", "lr": 0.05},
       "jiang": {}, "compute": {}}
N = 3


def _data():
    r = np.random.default_rng(0)
    xy = lambda n: (r.normal(size=(n, 6)).astype(np.float32), r.integers(0, 3, n))
    clients = []
    for c in range(N):
        X, y = xy(200)
        Xt, yt = xy(60)
        clients.append(Client(c, "general", X, y, Xt, yt, {}))
    (Xv, yv), (Xs, ys), (Xp, yp) = xy(80), xy(80), xy(30)
    return FedData(["Normal", "A", "B"], [f"f{i}" for i in range(6)], clients, Xv, yv, Xs, ys, Xp, yp, 2, 0, {})


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("shared") / "part")
    save_shared(_data(), d, False)
    return d


def test_mtls_pool_identical(shared, tmp_path):
    sim = FedSim(CFG, _data(), 1)
    out = str(tmp_path)
    pool = mtls.MTLSPool(CFG, 1, shared, "fedavg", range(N), str(tmp_path / "pki"), out, log=lambda m: None)
    try:
        g_ser = g_pool = sim.init_vec.clone()
        for rnd in range(3):
            torch.manual_seed(123 + rnd)  # parent RNG before the round; both paths must leave the same state
            ser = [sim.local_train(g_ser, c, rnd) for c in range(N)]
            rng_ser = torch.get_rng_state()
            torch.manual_seed(123 + rnd)
            done = []
            par = pool.train(g_pool, list(range(N)), rnd, on_done=lambda c, ms: done.append(c))
            assert torch.equal(torch.get_rng_state(), rng_ser)
            assert sorted(done) == list(range(N))
            assert all(torch.equal(a, b) for a, b in zip(ser, par))
            g_ser, g_pool = torch.stack(ser).mean(0), torch.stack(par).mean(0)
            for kind in ("voter_score", "voter_score_se"):
                assert pool.scores(kind, g_pool, [2, 0]) == {c: getattr(sim, kind)(g_ser, c) for c in (2, 0)}
    finally:
        pool.close()
    meta = json.load(open(os.path.join(out, "mtls_meta.json")))
    assert meta["status"] == "closed" and len(meta["drones"]) == N
    assert all(d["status"] == "connected" and d["tls_version"] == "TLSv1.3" and d["cipher"] for d in meta["drones"])
    assert all(v == 0 for v in meta["exit_codes"].values())
    lines = [json.loads(l) for l in open(os.path.join(out, "mtls.jsonl"))]
    assert [x["round"] for x in lines] == [0, 1, 2]
    assert all(d["bytes_up"] > 0 and d["bytes_down"] > 0 and d["tasks"] >= 1 for x in lines for d in x["drones"])
    assert json.load(open(os.path.join(out, "live_state.json")))["phase"] == "done"
    assert not any(p.is_alive() for p in pool.procs.values())


def test_revoked_drone_refused(shared, tmp_path):
    pki = str(tmp_path / "pki")
    mtls.ensure_pki(pki, N)
    serial = mtls.set_revoked(pki, 1)[0]
    assert mtls.pki_info(pki, N)["drones"][1]["revoked"]
    with pytest.raises(RuntimeError, match="UNAUTHENTICATED.*drone-02.*revoked"):
        mtls.MTLSPool(CFG, 1, shared, "fedavg", range(N), pki, str(tmp_path), log=lambda m: None)
    meta = json.load(open(os.path.join(tmp_path, "mtls_meta.json")))
    d = meta["drones"][1]
    assert d["status"] == "refused" and d["serial"] == serial and "revoked" in d["reason"]
    assert all(x["status"] != "refused" for i, x in enumerate(meta["drones"]) if i != 1)
    assert mtls.set_revoked(pki, 1, revoked=False) == []  # restore
