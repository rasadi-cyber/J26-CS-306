"""DQN for aggregation-weight selection of voted-in models (Jiang et al. 2024, Sec. 4).

State  (Sec. 4): [FL epoch r / R, voting freshness r'/r, accuracy of the target model, yes-vote share]
Action: one weight from a discrete list (the paper uses discrete weights)
Reward (Eq. 28): Acc((1 - w) * M_G + w * M_target) - Acc(M_G), measured on the public exam
Learning: standard DQN - experience replay, target network, epsilon-greedy, Huber loss. The paper's
"reward of predicted next state" is taken as the usual discounted max-Q of the next decision's state.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

STATE_DIM = 4


class QNet(nn.Module):
    def __init__(self, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.actions = list(cfg["actions"])
        self.rng = random.Random(seed)
        g = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng():
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
            self.q = QNet(len(self.actions), cfg["hidden"])
        self.target = QNet(len(self.actions), cfg["hidden"])
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buffer = deque(maxlen=cfg["buffer_size"])
        self.decisions = 0
        self.eps_override = None
        self.frozen = False  # pre-trained and used as-is (Jiang: pre-trained DRL vs DRL "that keeps training")
        self.pending = None  # last (state, action, reward) waiting for its next state
        self.last_loss = None
        # Running mean / variance of the state features (Welford). The raw features sit in narrow, different
        # ranges (accuracy 0.70-0.80, freshness ~0.99), so the Q-network sees them standardised.
        self.norm = cfg.get("normalize_state", False)
        self.s_n, self.s_mean, self.s_m2 = 0, np.zeros(STATE_DIM), np.zeros(STATE_DIM)

    # ----- persistence (fresh vs carried-over agent) -----
    def state_dict(self):
        return {"q": self.q.state_dict(), "target": self.target.state_dict(), "opt": self.opt.state_dict(),
                "buffer": list(self.buffer), "decisions": self.decisions, "actions": self.actions,
                "state_stats": (self.s_n, self.s_mean.tolist(), self.s_m2.tolist())}

    def load_state_dict(self, sd):
        if list(sd["actions"]) != self.actions:
            raise ValueError("saved DQN uses a different action list")
        self.q.load_state_dict(sd["q"])
        self.target.load_state_dict(sd["target"])
        self.opt.load_state_dict(sd["opt"])
        self.buffer.extend(sd["buffer"])
        self.decisions = sd["decisions"]
        if "state_stats" in sd:
            n, mean, m2 = sd["state_stats"]
            self.s_n, self.s_mean, self.s_m2 = n, np.array(mean), np.array(m2)

    def _observe(self, state):
        x = np.asarray(state, dtype=np.float64)
        self.s_n += 1
        d = x - self.s_mean
        self.s_mean += d / self.s_n
        self.s_m2 += d * (x - self.s_mean)

    def _scale(self, states):
        x = np.asarray(states, dtype=np.float64)
        if not self.norm or self.s_n < 2:
            return x
        return (x - self.s_mean) / np.sqrt(self.s_m2 / (self.s_n - 1) + 1e-8)

    # ----- acting / learning -----
    def epsilon(self):
        if self.eps_override is not None:
            return self.eps_override
        c = self.cfg
        frac = min(1.0, self.decisions / max(1, c["eps_decay_decisions"]))
        return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])

    def act(self, state, allowed=None):
        """allowed: indices of the actions permitted here (e.g. no weight 0 for a model the vote readmitted)."""
        if self.norm and not self.frozen:
            self._observe(state)
        eps = 0.0 if self.frozen else self.epsilon()
        allowed = list(range(len(self.actions))) if allowed is None else list(allowed)
        if self.rng.random() < eps:
            a = self.rng.choice(allowed)
        else:
            with torch.no_grad():
                q = self.q(torch.tensor(self._scale(state), dtype=torch.float32))
            a = allowed[int(q[allowed].argmax())]
        return a, eps

    def reset_for_new_setup(self):
        """Keep the learned network, drop what belongs to the old setup: replay memory (old rewards) and the
        input normalisation statistics. Used when a DQN is carried across versions (transfer, then fine-tune)."""
        self.buffer.clear()
        self.pending = None
        self.s_n, self.s_mean, self.s_m2 = 0, np.zeros(STATE_DIM), np.zeros(STATE_DIM)

    def record(self, state, action, reward):
        """Store a finished decision; the previous decision gets this state as its next state."""
        if self.frozen:
            return
        if self.pending is not None:
            s, a, r = self.pending
            self.buffer.append((s, a, r, list(state)))
        self.pending = (list(state), action, reward)
        self.decisions += 1
        self._learn()
        if self.decisions % self.cfg["target_sync_decisions"] == 0:
            self.target.load_state_dict(self.q.state_dict())

    def _learn(self):
        c = self.cfg
        if len(self.buffer) < c["batch_size"]:
            return
        for _ in range(c["train_steps"]):
            batch = self.rng.sample(list(self.buffer), c["batch_size"])
            s, a, r, s2 = zip(*batch)
            s, s2 = (torch.tensor(self._scale(np.array(x)), dtype=torch.float32) for x in (s, s2))
            a, r = torch.tensor(a), torch.tensor(r, dtype=torch.float32)
            q = self.q(s).gather(1, a.long().unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target = r + c["gamma"] * self.target(s2).max(1).values
            loss = F.smooth_l1_loss(q, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.last_loss = loss.item()
