"""Sanity tests: run with  .venv/bin/python -m pytest -q tests"""
import json
import os

import numpy as np
import torch

from fldos.aggregators import Jiang, Update
from fldos.model import MLP, get_vector, set_vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_set_vector_copies():
    m = MLP(5, 3)
    v = get_vector(m) + 1.0
    before = v.clone()
    set_vector(m, v)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)  # training the model must not touch the caller's vector
    assert torch.equal(v, before)


class _FakeSim:
    """Voters score a model by how close it is to a 'good' vector (higher = better)."""
    def __init__(self, good):
        self.good = good

    def voter_score(self, vec, cid):
        return -float((vec - self.good).norm())

    def voter_score_se(self, vec, cid):
        return self.voter_score(vec, cid), 0.01

    def voter_scores(self, vec, cids, kind="voter_score"):  # as FedSim.voter_scores (serial path)
        return {c: getattr(self, kind)(vec, c) for c in cids}

    def public_score(self, vec):
        return -float((vec - self.good).norm())


def _cfg():
    with open(os.path.join(ROOT, "configs", "default.json")) as f:
        cfg = json.load(f)
    cfg["training"]["rounds"] = 10
    cfg["jiang"]["vote_start_frac"] = 0.0
    return cfg


def test_filter_drops_outliers_and_vote_rejects_bad_model():
    torch.manual_seed(0)
    good = torch.zeros(20)
    sim = _FakeSim(good)
    J = Jiang(_cfg(), sim, seed=0, weighting="fixed")
    ups = [Update(i, "general", good + 0.01 * torch.randn(20), 100) for i in range(8)]
    ups.append(Update(8, "rare", good + 0.5 * torch.randn(20), 100))       # far but still near "good"
    ups.append(Update(9, "poisoned", good + 5.0 + torch.randn(20), 100))   # far and bad
    g = good.clone()
    for rnd in range(3):
        g, rows, _ = J.aggregate(rnd, ups, g)
    by = {r["cid"]: r for r in rows}
    assert by[9]["filtered_out"] and by[8]["filtered_out"]
    assert by[9]["status"] == "rejected"          # poisoned: flagged, voted out


def test_vote_uses_strict_majority():
    sim = _FakeSim(torch.zeros(4))
    J = Jiang(_cfg(), sim, seed=0, weighting="fixed")
    voters = [Update(i, "general", torch.zeros(4), 1) for i in range(10)]
    # all 10 honest voters see an identical score -> ">=" gives 10 yes votes -> readmitted
    assert J._vote(torch.zeros(4), {i: (0.0, 0.0) for i in range(10)}, voters) == 10


def test_vote_margin_is_one_standard_error():
    cfg = _cfg()
    cfg["jiang"]["vote_margin_se"] = 1.0
    J = Jiang(cfg, _FakeSim(torch.zeros(4)), seed=0, weighting="fixed")
    voters = [Update(i, "general", torch.zeros(4), 1) for i in range(10)]
    near = torch.tensor([0.005, 0, 0, 0])   # score -0.005: worse than global (0.0) but within 1 SE (0.01)
    far = torch.tensor([0.5, 0, 0, 0])      # score -0.5: far outside the noise
    assert J._vote(near, {i: (0.0, 0.01) for i in range(10)}, voters) == 10
    assert J._vote(far, {i: (0.0, 0.01) for i in range(10)}, voters) == 0


def test_window_features_are_causal_and_per_stream():
    from fldos.data import window_features
    feats = ["frame.len", "frame.time_delta_displayed"]
    X = np.array([[10, 1], [20, 1], [30, 1], [100, 5], [100, 5]], dtype=float)
    labels = np.array(["A", "A", "A", "B", "B"])
    W, names = window_features(X, labels, feats, [2])
    m = W[:, names.index("win2.len_mean")]
    assert m[0] == 10 and m[1] == 15 and m[2] == 25      # only past packets of its own stream
    assert m[3] == 100                                     # a new stream does not see stream A


def test_model_poisoning_and_no_flip():
    from fldos.data import flip_labels
    from fldos.runner import poison_update
    cfg = {"clients": {"model_attack": "scale", "model_attack_factor": 10.0}}
    g, u = torch.zeros(3), torch.ones(3)
    assert torch.allclose(poison_update(cfg, "poisoned", g, u), torch.full((3,), 10.0))
    assert torch.equal(poison_update(cfg, "general", g, u), u)   # honest drones untouched
    cfg["clients"].update(model_attack="sign_flip", model_attack_factor=1.0)
    assert torch.allclose(poison_update(cfg, "poisoned", g, u), -torch.ones(3))
    y = np.array([0, 1, 2])
    assert np.array_equal(flip_labels(y, "none", 0, 3), y)
    assert np.array_equal(flip_labels(y, "attack_to_normal", 0, 3), np.zeros(3))


def test_prior_weighted_f1_charges_false_alarms_on_big_classes():
    from fldos.sim import macro_f1, macro_f1_prior
    y = np.array([0] * 10 + [1] * 10)                      # balanced exam
    pred = np.array([0] * 8 + [1] * 2 + [1] * 10)          # 2 of the big class flagged as the small one
    plain = macro_f1(y, pred, 2)
    natural = macro_f1_prior(y, pred, 2, [0.99, 0.01])    # class 0 is 99% of real traffic
    assert natural < plain                                  # the false alarms weigh more in natural traffic


def test_colluding_voters_back_each_other():
    cfg = _cfg()
    cfg["clients"]["malicious_vote"] = "collude"
    J = Jiang(cfg, _FakeSim(torch.zeros(4)), seed=0, weighting="fixed")
    voters = [Update(i, "poisoned" if i < 3 else "general", torch.zeros(4), 1) for i in range(10)]
    far = torch.tensor([5.0, 0, 0, 0])
    scores = {i: (0.0, 0.01) for i in range(10)}
    assert J._vote(far, scores, voters, owner_role="poisoned") == 3      # only the 3 colluders say yes
    assert J._vote(torch.zeros(4), scores, voters, owner_role="general") == 7  # colluders vote honest models out


def test_mixed_traffic_episodes_and_period_labels():
    from fldos.traffic import build
    rng = np.random.default_rng(0)
    n_norm, n_att = 300, 250
    length = np.r_[np.full(n_norm, 500.0), np.full(n_att, 54.0)]
    gap = np.r_[np.full(n_norm, 0.02), np.full(n_att, 1e-5)]
    cls = np.r_[np.zeros(n_norm, int), np.ones(n_att, int)]
    X, y, info = build(length, gap, cls, np.arange(len(cls)), 0, rng, [10], burst=(250, 250), bg_ratio=(1, 1))
    assert X.shape[1] == 2 + 9                       # len, gap + 9 window stats for one window size
    assert info["attack_packet"].sum() == n_att       # every attack packet used once
    att = info["attack_packet"]
    # deep inside the burst the link is labelled as under attack; well before it, normal
    first = np.flatnonzero(att)[0]
    assert y[first + 50] == 1 and y[: max(1, first - 20)].max() == 0


def test_rolling_folds_are_forward_and_rotate_normal_captures():
    import pandas as pd
    from fldos.data import fold_segments, ORDER_COL, SOURCE_COL, CAPTURE_COL
    n = 1000
    df = pd.DataFrame({ORDER_COL: np.arange(n), SOURCE_COL: ["DDoS"] * n, CAPTURE_COL: [0] * n})
    for fold in range(5):
        seg = fold_segments(df, fold, 0.2, 0.1)
        tr, te = np.flatnonzero(seg == 0), np.flatnonzero(seg == 2)
        assert tr.max() < te.min()                     # test is always later than training
    nm = pd.DataFrame({ORDER_COL: np.arange(500), SOURCE_COL: ["Normal"] * 500, CAPTURE_COL: np.repeat(range(5), 100)})
    tests = {tuple(np.unique(nm[CAPTURE_COL][fold_segments(nm, f, 0.2, 0.1) == 2])) for f in range(5)}
    assert len(tests) == 5                             # each fold tests on a different Normal capture


def test_dqn_cannot_zero_a_readmitted_model():
    from fldos.dqn import DQNAgent
    cfg = _cfg()["jiang"]["dqn"]
    agent = DQNAgent(cfg, 0)
    allowed = [i for i, w in enumerate(agent.actions) if w > 0]
    picks = {agent.act([0.5, 1.0, 0.9, 0.8], allowed)[0] for _ in range(200)}
    assert picks and all(agent.actions[a] > 0 for a in picks)
