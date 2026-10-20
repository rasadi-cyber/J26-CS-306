"""Mixed-traffic timelines: normal traffic with DoS / attack bursts interleaved in time, as a drone's link sees it.

Used to build every training, voting, validation and test set when `data.traffic = mixed`. Each episode:
  background  a run of normal packets (on their own inter-arrival timeline)
  burst       a run of one attack type's packets, starting at a random time inside the background
The two are merged by arrival time, inter-arrival gaps are recomputed over the merged stream, and the window
statistics are computed over that stream, so a window can hold both normal and attack packets.

Labels follow how link-level DoS detectors are judged: each packet is labelled with what the link is doing at that
moment - the attack type if attack packets make up at least `period_share` of the last `period_len` packets,
otherwise Normal. A normal packet in the middle of a flood is part of the attack period; one after the burst has
passed is normal again.

Only the rows of one split (one drone's training rows, its local test rows, validation or test) go into that
split's episodes, so no packet is shared between splits.
"""
import numpy as np

from .data import window_features


def _bursts(n, rng, lo, hi):
    """Cut n packets (in capture order) into consecutive bursts of random length in [lo, hi]."""
    cuts, i = [], 0
    while i < n:
        b = int(rng.integers(lo, hi + 1))
        cuts.append((i, min(n, i + b)))
        i += b
    return cuts


def _plan(cls, normal, n_normal, rng, burst, bg_ratio, period_len):
    """No-reuse plan: every burst's wanted background, scaled down so the split's normal packets are each used once."""
    plan = []
    for c in [c for c in np.unique(cls) if c != normal]:
        for s, e in _bursts(int((cls == c).sum()), rng, *burst):
            plan.append((c, s, e, max(period_len * 2, int((e - s) * rng.uniform(*bg_ratio)))))
    want = sum(p[3] for p in plan)
    if want > n_normal:  # spread the available normal traffic over all bursts in proportion to what each wanted
        edges = np.floor(np.cumsum([0] + [p[3] for p in plan]) * n_normal / want).astype(int)
        plan = [(c, s, e, int(edges[i + 1] - edges[i])) for i, (c, s, e, _) in enumerate(plan)]
    return plan


def build(length, gap, cls, order, normal, rng, window_sizes, burst=(200, 2000), bg_ratio=(0.5, 2.0),
          period_len=10, period_share=0.5, reuse_background=True, extra=None, extra_cols=None):
    """length, gap, cls, order: per-row arrays of one split. Returns (X raw features, period label, info).
    reuse_background=False: each normal packet is used once (no looping); bursts get less background when normal
    traffic is scarce, instead of the same few hundred normal packets being replayed hundreds of times.
    extra (rows x k, optional): per-packet protocol fields carried with each packet through the merge; their window
    means over the merged stream are appended (data.proto_path)."""
    o = np.argsort(order, kind="stable")
    length, gap, cls = length[o], np.clip(gap[o], 0, None), cls[o]
    is_n = cls == normal
    n_len, n_gap = length[is_n], gap[is_n]
    E = []
    if extra is not None:
        extra = extra[o]
        n_ex = extra[is_n]
    if len(n_len) == 0:
        raise ValueError("split has no normal traffic for a background")
    ptr = 0
    X, Y, EP, ATT, T = [], [], [], [], []
    ep = 0
    if reuse_background:  # v10: bursts drawn class by class (same random order as before)
        plan = ((c, s, e, None) for c in [c for c in np.unique(cls) if c != normal]
                for s, e in _bursts(int((cls == c).sum()), rng, *burst))
    else:
        plan = _plan(cls, normal, len(n_len), rng, burst, bg_ratio, period_len)
    for c, s, e, m_fixed in plan:
        a_len, a_gap = length[cls == c], gap[cls == c]
        b_len, b_gap = a_len[s:e], a_gap[s:e]
        if m_fixed is None:
            m = max(period_len * 2, int(len(b_len) * rng.uniform(*bg_ratio)))
            take = (ptr + np.arange(m)) % len(n_len)  # background reused cyclically if normal traffic is scarce
            ptr = (ptr + m) % len(n_len)
        else:
            m = min(m_fixed, len(n_len) - ptr)
            take = ptr + np.arange(m)
            ptr += m
        bg_len, bg_gap = n_len[take], n_gap[take]
        if extra is not None:
            ex = np.vstack([n_ex[take], extra[cls == c][s:e]])
        t_bg = np.cumsum(bg_gap)
        start = rng.uniform(0.1, 0.6) * max(t_bg[-1] if m else 0.0, 1e-6)
        t_b = start + np.cumsum(b_gap)
        t = np.concatenate([t_bg, t_b])
        ln = np.concatenate([bg_len, b_len])
        att = np.concatenate([np.zeros(m, bool), np.ones(len(b_len), bool)])
        srt = np.argsort(t, kind="stable")
        t, ln, att = t[srt], ln[srt], att[srt]
        if extra is not None:
            E.append(ex[srt])
        g = np.diff(t, prepend=t[0])
        # period label: share of attack packets among the last period_len packets
        cs = np.cumsum(att)
        win = np.minimum(np.arange(1, len(att) + 1), period_len)
        share = (cs - np.concatenate([np.zeros(period_len), cs[:-period_len]])[:len(cs)]) / win
        y = np.where(share >= period_share, c, normal)
        X.append(np.column_stack([ln, g])); Y.append(y); EP.append(np.full(len(t), ep)); ATT.append(att); T.append(t)
        ep += 1
    if ep == 0:  # normal traffic only (e.g. a voter with no attack rows)
        X, Y, EP, ATT, T = [np.column_stack([n_len, n_gap])], [np.full(len(n_len), normal)], [np.zeros(len(n_len), int)], \
            [np.zeros(len(n_len), bool)], [np.cumsum(n_gap)]
        E = [n_ex] if extra is not None else []
    X, Y, EP, ATT, T = (np.concatenate(a) if a[0].ndim == 1 else np.vstack(a) for a in (X, Y, EP, ATT, T))
    W, _ = window_features(X, EP, ["frame.len", "frame.time_delta_displayed"], window_sizes, full=True)
    out = np.hstack([X, W])
    if extra is not None:
        from .data import proto_window_features
        E = np.vstack(E).astype(np.float32)
        PW, _ = proto_window_features(E, list(extra_cols), EP, window_sizes)
        out = np.hstack([out, E, PW])
    return out, Y, {"episode": EP, "attack_packet": ATT, "time": T}
