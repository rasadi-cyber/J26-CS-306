"""Server-side aggregation rules.

fedavg        FedAvg (McMahan et al.): data-size weighted mean of every update.
fedavg_clean  Reference only: FedAvg with the poisoned drone removed by an oracle (the "no attacker" line).
multi_krum    Multi-Krum (Blanchard et al.): Krum score over the n - f - 2 nearest updates, average the K best.
jiang_fixed   Jiang et al. filtering + voting with a fixed weight 1/(K+1) for voted-in models (paper's ablation).
jiang_dqn     Jiang et al. filtering + voting + DQN weight selection (online: fresh agent each run).
jiang_dqn_pre Same, starting from a DQN carried over from earlier tuning runs (paper's pre-trained DRL).
fedwgca       Robustness-side baseline, FedWGCA (Fahim-Ul-Islam et al., IEEE OJCS 2025, Sec. V-G / Algorithm 1):
              drop updates whose norm is outside [Q1 - lam*IQR, Q3 + lam*IQR], clip each value to [-C, C], weight
              each kept update by 1 / ||clipped update|| (normalised), global = global + sum w_i * clipped update.
fed_grab      Rare-class-side baseline, Fed-GraB (Xiao et al., NeurIPS 2023): every drone trains with the
              self-adjusting gradient balancer (PID-controlled positive/negative gradient weights, prior from the
              global classifier's weight norms, see sim.py); the server is plain FedAvg (as in the authors' code).

Jiang et al. (2024), Algorithm 1:
  filter   score_i = sum_x ||M_i - M_x||^2 over all drones; the K smallest form S; M_B = |D|-weighted mean of S
  vote     (lagged by vote_lag rounds) every drone tests each dropped model M_U^{r'} and the global M_G^{r'} on its
           own test set; votes 1 if Acc(M_U) >= Acc(M_G). Readmitted if yes > N/2. Malicious drones vote randomly.
           vote_margin_se = z reads ">=" as "not worse by more than z standard errors of Acc(M_G)" (README).
  combine  M_G^r = (M_B^r + sum_q w_q M_V^{r'}) / (1 + sum_q w_q), or M_B^r when nothing is voted in.
The code never reads drone roles, except fedavg_clean (reference) and the malicious drones' own random votes.
"""
import random
from dataclasses import dataclass

import numpy as np
import torch

from .dqn import DQNAgent


@dataclass
class Update:
    cid: int
    role: str
    vec: torch.Tensor
    n: int


def wavg(vecs, weights):
    w = torch.tensor(weights, dtype=torch.float32, device=vecs[0].device)
    return (torch.stack(vecs) * (w / w.sum())[:, None]).sum(0)


def _row(u, status, share, **extra):
    return {"cid": u.cid, "role": u.role, "n": u.n, "status": status, "weight": float(share), **extra}


class FedAvg:
    name = "fedavg"

    def __init__(self, cfg, sim, seed):
        self.cfg, self.sim = cfg, sim

    def keep(self, u):
        return True

    def aggregate(self, rnd, updates, global_vec):
        kept = [u for u in updates if self.keep(u)]
        total = sum(u.n for u in kept)
        new = wavg([u.vec for u in kept], [u.n for u in kept])
        rows = [_row(u, "averaged" if self.keep(u) else "excluded", u.n / total if self.keep(u) else 0.0)
                for u in updates]
        return new, rows, {}


class FedAvgClean(FedAvg):
    name = "fedavg_clean"

    def keep(self, u):
        return u.role != "poisoned"  # oracle: reference line only


class MultiKrum:
    name = "multi_krum"

    def __init__(self, cfg, sim, seed):
        self.K, self.f = cfg["multi_krum"]["K"], cfg["multi_krum"]["f"]

    def aggregate(self, rnd, updates, global_vec):
        V = torch.stack([u.vec for u in updates])
        D = torch.cdist(V, V) ** 2
        m = max(1, len(updates) - self.f - 2)
        scores = torch.sort(D, dim=1).values[:, 1:m + 1].sum(1).tolist()
        sel = sorted(range(len(updates)), key=lambda i: scores[i])[:self.K]
        total = sum(updates[i].n for i in sel)
        new = wavg([updates[i].vec for i in sel], [updates[i].n for i in sel])
        rows = [_row(u, "accepted" if i in sel else "flagged", u.n / total if i in sel else 0.0, score=scores[i])
                for i, u in enumerate(updates)]
        return new, rows, {}


class Jiang:
    name = "jiang"

    def __init__(self, cfg, sim, seed, weighting="dqn", agent=None):
        jc = cfg["jiang"]
        self.cfg, self.sim, self.jc = cfg, sim, jc
        self.K, self.lag = jc["K"], jc["vote_lag"]
        self.R = cfg["training"]["rounds"]
        self.vote_start = int(round(jc["vote_start_frac"] * self.R))
        self.weighting = weighting
        self.fixed_w = jc.get("fixed_weight") or 1.0 / (self.K + 1)
        self.agent = agent if agent is not None else (DQNAgent(jc["dqn"], seed) if weighting == "dqn" else None)
        self.vote_margin = jc.get("vote_margin_se", 0.0)
        self.vote_rng = random.Random(seed + 7)
        self.dropped = {}   # L_U: round -> dropped updates
        self.globals = {}   # L_G: round -> global model of that round

    # --- Algorithm 1, filtering layer ---
    def _filter(self, updates):
        V = torch.stack([u.vec for u in updates])
        scores = (torch.cdist(V, V) ** 2).sum(1).tolist()
        order = sorted(range(len(updates)), key=lambda i: scores[i])
        return order[:self.K], order[self.K:], scores

    # --- Algorithm 1, voting layer ---
    def _vote(self, model_vec, global_scores, voters, owner_role=None):
        yes = 0
        mv = self.cfg["clients"]["malicious_vote"]
        honest = [v.cid for v in voters if not (v.role == "poisoned" and mv in ("random", "collude"))]
        score = self.sim.voter_scores(model_vec, honest)  # computed up front (in parallel when enabled)
        for v in voters:
            if v.role == "poisoned" and self.cfg["clients"]["malicious_vote"] == "random":
                yes += self.vote_rng.random() < self.cfg["clients"]["malicious_vote_p"]
            elif v.role == "poisoned" and self.cfg["clients"]["malicious_vote"] == "collude":
                yes += owner_role == "poisoned"  # attackers back each other and vote every honest model out
            else:
                acc_g, se_g = global_scores[v.cid]
                yes += score[v.cid] >= acc_g - self.vote_margin * se_g
        return yes

    def aggregate(self, rnd, updates, global_vec):
        sel, drop, scores = self._filter(updates)
        total_sel = sum(updates[i].n for i in sel)
        M_B = wavg([updates[i].vec for i in sel], [updates[i].n for i in sel])
        self.dropped[rnd] = [updates[i] for i in drop]
        N = len(updates)

        # Voting runs `lag` rounds behind the filter and is frozen at the start of training.
        r_prime = rnd - self.lag
        voted = {}  # cid -> dict(yes, readmitted, weight, ...)
        info = {}
        # vote_lag = 0 (synchronous voting, the paper's lag bound e with instant communication): this round's
        # dropped models are judged against this round's filtered model M_B, the global before readmission.
        ref = M_B if self.lag == 0 else self.globals.get(r_prime)
        if rnd >= self.vote_start and r_prime in self.dropped and ref is not None:
            g_scores = self.sim.voter_scores(ref, [v.cid for v in updates], kind="voter_score_se")
            for u in self.dropped.pop(r_prime):
                yes = self._vote(u.vec, g_scores, updates, u.role)
                voted[u.cid] = {"u": u, "yes": int(yes), "readmitted": yes > N / 2}
        for r in [r for r in self.dropped if r < rnd - self.lag]:
            del self.dropped[r]

        readmitted = [v for v in voted.values() if v["readmitted"]]
        mode = self.jc["dqn"].get("reward_mix", "marginal")
        # The DQN can also weigh the filter's accepted models (proposal: "DQN selects aggregation weight per
        # accepted/readmitted update"); otherwise accepted models keep their plain |D| weights inside M_B.
        acc_w = {i: 1.0 for i in sel}
        accepted = []
        if self.weighting == "dqn" and self.jc["dqn"].get("weight_accepted") and rnd >= self.vote_start:
            for i in sel:
                state = [rnd / self.R, 1.0, self.sim.public_score(updates[i].vec), 1.0]
                # accepted_positive_only: an accepted model may be re-weighted but not dropped (as for readmitted ones)
                allowed = ([k for k, w in enumerate(self.agent.actions) if w > 0]
                           if self.jc["dqn"].get("accepted_positive_only") else None)
                a, eps = self.agent.act(state, allowed)
                acc_w[i] = self.agent.actions[a]
                accepted.append({"i": i, "u": updates[i], "weight": acc_w[i], "action": a, "state": state, "eps": eps})

        def m_b(w):
            ws = [updates[i].n * w[i] for i in sel]
            return wavg([updates[i].vec for i in sel], ws) if sum(ws) > 0 else M_B

        M_Bw = m_b(acc_w)
        for v in readmitted:
            if self.weighting == "fixed":
                v["weight"] = self.fixed_w
                continue
            state = [rnd / self.R, r_prime / max(1, rnd), self.sim.public_score(v["u"].vec), v["yes"] / N]
            # The vote readmitted this model: the DQN sets how much it counts, but may not silently drop it (w > 0).
            allowed = ([i for i, w in enumerate(self.agent.actions) if w > 0]
                       if self.jc["dqn"].get("readmit_positive_only") else None)
            a, eps = self.agent.act(state, allowed)
            # fixed_multiple: the action scales the paper's fixed weight 1/(K+1) (action 1 = the fixed-weight method)
            scale = self.fixed_w if self.jc["dqn"].get("action_scale") == "fixed_multiple" else 1.0
            v.update(weight=scale * self.agent.actions[a], action=a, state=state, eps=eps)

        def eq9(models, base=None):
            base = M_Bw if base is None else base
            sw = sum(m["weight"] for m in models)
            return (base + sum(m["weight"] * m["u"].vec for m in models)) / (1 + sw) if sw > 0 else base

        new = eq9(readmitted)  # Eq. 9
        if self.weighting == "dqn" and (readmitted or accepted):
            # Reward = accuracy gain on the public exam from the target model at its chosen weight.
            #   eq28        paper's literal mix (1-w)M_G + wM, each model on its own
            #   aggregation Eq. 9 mix of M_B and this model alone
            #   marginal    Eq. 9 with every voted-in model vs. the same without this one (its real contribution)
            acc_new = self.sim.public_score(new) if mode == "marginal" else None
            acc_base = self.sim.public_score(M_B) if mode in ("eq28", "aggregation") else None
            default_w = self.fixed_w if self.jc["dqn"].get("action_scale") == "fixed_multiple" else 1.0
            for v in readmitted:
                w, x = v["weight"], v["u"].vec
                if mode == "vs_default":
                    # Gain over giving this model the default weight (the fixed-weight method's choice), with every
                    # other decision unchanged: the DQN is only rewarded for doing better than the fixed weight.
                    if w == default_w:
                        reward = 0.0
                    else:
                        if acc_new is None:
                            acc_new = self.sim.public_score(new)
                        alt = [dict(m, weight=default_w) if m is v else m for m in readmitted]
                        reward = acc_new - self.sim.public_score(eq9(alt))
                elif w == 0:
                    reward = 0.0
                elif mode == "eq28":
                    reward = self.sim.public_score((1 - w) * M_B + w * x) - acc_base
                elif mode == "aggregation":
                    reward = self.sim.public_score((M_B + w * x) / (1 + w)) - acc_base
                else:
                    reward = acc_new - self.sim.public_score(eq9([m for m in readmitted if m is not v]))
                self.agent.record(v["state"], v["action"], reward)
                v.update(reward=round(reward, 4), state=[round(z, 4) for z in v["state"]], eps=round(v["eps"], 3))
            for v in accepted:
                if mode == "vs_default":
                    if v["weight"] == 1.0:
                        reward = 0.0
                    else:
                        if acc_new is None:
                            acc_new = self.sim.public_score(new)
                        reward = acc_new - self.sim.public_score(eq9(readmitted, m_b({**acc_w, v["i"]: 1.0})))
                elif v["weight"] == 0:
                    reward = 0.0
                else:
                    if acc_new is None:
                        acc_new = self.sim.public_score(new)
                    reward = acc_new - self.sim.public_score(eq9(readmitted, m_b({**acc_w, v["i"]: 0.0})))
                self.agent.record(v["state"], v["action"], reward)
                v.update(reward=round(reward, 4), state=[round(z, 4) for z in v["state"]], eps=round(v["eps"], 3))
        acc_dec = {v["i"]: v for v in accepted}
        total_selw = sum(updates[i].n * acc_w[i] for i in sel)
        if total_selw == 0:  # every accepted model got weight 0: M_B fell back to plain |D| weights
            acc_dec, total_selw = {}, total_sel
        sum_w = sum(v["weight"] for v in readmitted)
        self.globals[rnd] = new
        for r in [r for r in self.globals if r < rnd - self.lag]:
            del self.globals[r]

        rows = []
        for i, u in enumerate(updates):
            share = (u.n * acc_w[i] / total_selw if acc_dec else u.n / total_sel) / (1 + sum_w) if i in sel else 0.0
            v = voted.get(u.cid)
            if v and v["readmitted"]:
                share += v["weight"] / (1 + sum_w)
            if v and v["readmitted"] and v["weight"] > 0:
                status = "readmitted"
            elif i in sel:
                status = "accepted"
            elif v and v["readmitted"]:
                status = "vetoed"      # voted in, but the DQN chose weight 0
            elif v:
                status = "rejected"    # its previous-round model was voted out
            else:
                status = "flagged"     # dropped by the filter; vote frozen or pending
            row = _row(u, status, share, score=scores[i], filtered_out=i not in sel)
            if v:
                row.update(votes_yes=v["yes"], votes_total=N, vote_round=r_prime, readmitted=v["readmitted"],
                           dqn_weight=v.get("weight"), reward=v.get("reward"), dqn_state=v.get("state"))
            elif i in acc_dec:
                a = acc_dec[i]
                row.update(dqn_weight=a["weight"], reward=a["reward"], dqn_state=a["state"])
            rows.append(row)
        if self.agent is not None:
            info = {"epsilon": round(self.agent.epsilon(), 4), "dqn_loss": self.agent.last_loss,
                    "dqn_decisions": self.agent.decisions}
        info["voting_active"] = rnd >= self.vote_start
        return new, rows, info


class FedWGCA:
    name = "fedwgca"

    def __init__(self, cfg, sim, seed):
        c = cfg.get("fedwgca", {})
        self.lam = c.get("iqr_lambda", 1.5)  # Tukey's fence (the paper calls lambda tunable without a value)
        self.C = c.get("clip", 1.0)          # the paper's clipping value (Sec. VI-D)
        self.q = c.get("quartile", 0.25)     # lower quantile for "Q1" (paper Table 4 ablates 0.10-0.35); Q3 = 1 - q

    def aggregate(self, rnd, updates, global_vec):
        deltas = [u.vec - global_vec for u in updates]
        norms = np.array([float(d.norm()) for d in deltas])
        q1, q3 = np.percentile(norms, [100 * self.q, 100 * (1 - self.q)])
        lo, hi = q1 - self.lam * (q3 - q1), q3 + self.lam * (q3 - q1)
        keep = [i for i in range(len(updates)) if lo <= norms[i] <= hi]
        clipped = {i: deltas[i].clamp(-self.C, self.C) for i in keep}
        inv = {i: 1.0 / max(float(clipped[i].norm()), 1e-12) for i in keep}
        tot = sum(inv.values())
        new = global_vec + sum(clipped[i] * (inv[i] / tot) for i in keep)
        rows = [_row(u, "accepted" if i in inv else "flagged", inv[i] / tot if i in inv else 0.0, score=float(norms[i]))
                for i, u in enumerate(updates)]
        return new, rows, {}


class RCSFed:
    """RCS-Fed (Stolz et al., IEEE AIIoT 2026, Sec. IV, Algorithm 1): rare-class contribution scoring.
    Each client sends its class-count histogram once; rarity rho_c = 1/N_c for attack classes (benign excluded),
    score s_k = sum_c n_kc / N_c (Eq. 5); weight w_k = s_k * ||theta_k - theta_{t-1}||^2, normalised (Eq. 6);
    global = sum_k w_k theta_k. No defence against poisoned updates (the paper's setting is benign).
    rcs_fed.counts: "train" = histogram of the labels each drone trains on (a poisoned drone reports its flipped
    labels); "true" = histogram of each drone's own rows (a poisoned drone reports honest-looking counts)."""
    name = "rcs_fed"

    def __init__(self, cfg, sim, seed, oracle_clean=False):
        self.cfg, self.sim, self.oracle_clean = cfg, sim, oracle_clean
        d = sim.data
        C, normal = len(d.class_names), d.normal_target
        mode = cfg.get("rcs_fed", {}).get("counts", "train")
        hist = {}
        for c in d.clients:
            if mode == "true":
                hist[c.cid] = np.array([c.counts.get(n, 0) for n in d.class_names], dtype=np.float64)
            else:
                hist[c.cid] = np.bincount(np.asarray(c.y_train), minlength=C).astype(np.float64)
        considered = [cid for cid in hist if not (oracle_clean and d.clients[cid].role == "poisoned")]
        N = sum(hist[k] for k in considered)
        rho = np.where(N > 0, 1.0 / np.maximum(N, 1), 0.0)
        rho[normal] = 0.0  # benign class excluded (Eq. 4)
        self.score = {k: float((hist[k] * rho).sum()) for k in hist}

    def keep(self, u):
        return not (self.oracle_clean and u.role == "poisoned")

    def aggregate(self, rnd, updates, global_vec):
        kept = [u for u in updates if self.keep(u)]
        raw = [self.score[u.cid] * float(((u.vec - global_vec) ** 2).sum()) for u in kept]
        if sum(raw) <= 0:  # degenerate round: fall back to data-size weights
            raw = [float(u.n) for u in kept]
        new = wavg([u.vec for u in kept], raw)
        tot = sum(raw)
        share = {u.cid: r / tot for u, r in zip(kept, raw)}
        rows = [_row(u, "averaged" if self.keep(u) else "excluded", share.get(u.cid, 0.0), rcs_score=round(self.score[u.cid], 4))
                for u in updates]
        return new, rows, {}


def build(method, cfg, sim, seed, agent=None):
    if method in ("fedrs", "fedlc"):
        return FedAvg(cfg, sim, seed)  # FedRS / FedLC change only the local loss; the server is FedAvg
    if method in ("fedrs_clean", "fedlc_clean"):
        return FedAvgClean(cfg, sim, seed)  # reference: the papers' benign setting
    if method == "rcs_fed":
        return RCSFed(cfg, sim, seed)
    if method == "rcs_fed_clean":
        return RCSFed(cfg, sim, seed, oracle_clean=True)  # reference: the paper's benign setting (no attackers)
    if method == "fedwgca":
        return FedWGCA(cfg, sim, seed)
    if method == "fed_grab":
        return FedAvg(cfg, sim, seed)  # Fed-GraB's server step is FedAvg; its method is the local balancer
    if method == "fedavg":
        return FedAvg(cfg, sim, seed)
    if method == "fedavg_clean":
        return FedAvgClean(cfg, sim, seed)
    if method == "multi_krum":
        return MultiKrum(cfg, sim, seed)
    if method == "jiang_fixed":
        return Jiang(cfg, sim, seed, weighting="fixed")
    if method in ("jiang_dqn", "jiang_dqn_pre", "jiang_dqn_prexv"):
        return Jiang(cfg, sim, seed, weighting="dqn", agent=agent)
    raise ValueError(method)


METHODS = ["fedavg_clean", "fedavg", "multi_krum", "fedwgca", "fed_grab", "jiang_fixed", "jiang_dqn", "jiang_dqn_pre",
           "jiang_dqn_prexv", "rcs_fed", "rcs_fed_clean", "fedrs", "fedrs_clean", "fedlc", "fedlc_clean"]
