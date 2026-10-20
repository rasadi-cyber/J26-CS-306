"""Single-machine federated simulation: drones' local training and the scoring hooks the server uses."""
import numpy as np
import torch
import warnings

# compute.shared_data: features are read-only memory maps; torch only reads them (batches are copied)
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
import torch.nn.functional as F

from .data import FedData
from .model import MLP, get_vector, set_vector


def balanced_accuracy(y, pred, n_classes):
    """Mean recall over the classes present in y (so a 1% class counts as much as a 50% class)."""
    present = np.bincount(y, minlength=n_classes) > 0
    hit = np.bincount(y[pred == y], minlength=n_classes)
    tot = np.bincount(y, minlength=n_classes)
    return float((hit[present] / tot[present]).mean())


def macro_f1(y, pred, n_classes):
    """Macro F1 over the classes present in y: unlike balanced accuracy it also charges false positives."""
    present = np.bincount(y, minlength=n_classes) > 0
    cm = np.zeros((n_classes, n_classes))
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n_classes), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n_classes), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n_classes), where=(prec + rec) > 0)
    return float(f1[present].mean())


def macro_f1_prior(y, pred, n_classes, prior):
    """Macro F1 with each true class's rows reweighted to its natural frequency (prior / count), so a balanced
    exam is scored like natural traffic: errors on big classes (false alarms) weigh what they would in deployment."""
    present = np.bincount(y, minlength=n_classes) > 0
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    cm = np.zeros((n_classes, n_classes))
    np.add.at(cm, (y, pred), 1)
    cm = cm * np.divide(np.asarray(prior, float), cnt, out=np.zeros(n_classes), where=cnt > 0)[:, None]
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n_classes), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n_classes), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n_classes), where=(prec + rec) > 0)
    return float(f1[present].mean())


def balanced_accuracy_se(y, pred, n_classes, groups=None):
    """Standard error of balanced accuracy: per-class recall is a binomial proportion, classes are independent.
    groups (cluster-robust SE): rows of one group (a traffic burst) are not independent - consecutive packets share
    their window statistics - so each class's recall is treated as a ratio estimate over groups (standard clustered
    variance, G/(G-1) * sum_g (hits_g - recall * n_g)^2 / n^2). With one row per group this equals the binomial SE."""
    present = np.bincount(y, minlength=n_classes) > 0
    tot = np.bincount(y, minlength=n_classes)[present]
    rec = np.bincount(y[pred == y], minlength=n_classes)[present] / tot
    if groups is None:
        return float(np.sqrt((rec * (1 - rec) / tot).sum()) / present.sum())
    var = []
    for k, r, n in zip(np.flatnonzero(present), rec, tot):
        m = y == k
        _, g = np.unique(groups[m], return_inverse=True)
        G = g.max() + 1
        if G < 2:
            var.append(r * (1 - r) / n)
            continue
        dev = np.bincount(g, weights=(pred[m] == k) - r, minlength=G)
        var.append(G / (G - 1) * (dev ** 2).sum() / n ** 2)
    return float(np.sqrt(np.sum(var)) / present.sum())


class FedSim:
    def __init__(self, cfg: dict, data: FedData, seed: int, device=None):
        tc = cfg["training"]
        self.cfg, self.data, self.seed = cfg, data, seed
        self.device = torch.device(device or tc["device"])  # device: per-job override (CPU + GPU scheduling)
        self.C = len(data.class_names)
        torch.manual_seed(seed)  # identical initial global model for every method of this seed
        self.model = MLP(data.n_features, self.C, tc["hidden"], tc["dropout"]).to(self.device)
        self.init_vec = get_vector(self.model)
        # features keep the partition's storage dtype (float32, or float16 with compute.feature_dtype); every
        # model call casts its batch to float32 (mixed-precision storage, full-precision compute)
        t = lambda a, dt=None: torch.as_tensor(a, dtype=dt, device=self.device)
        self.train_t = [(t(c.X_train), t(c.y_train, torch.long)) for c in data.clients]
        self.vote_X = [t(c.X_test) for c in data.clients]
        self.vote_y = [c.y_test for c in data.clients]
        self.public = (t(data.X_public), data.y_public)
        self.val = (t(data.X_val), data.y_val)
        self.test = (t(data.X_test), data.y_test)
        self.test_pure = (t(data.X_test_pure), data.y_test_pure) if getattr(data, "X_test_pure", None) is not None else None
        if self.device.type != "cpu":  # GPU job: the tensors are copies, so release the host arrays (one copy in RAM)
            for c in data.clients:
                c.X_train = c.X_test = None
            for k in ("X_public", "X_val", "X_test", "X_test_pure"):
                if getattr(data, k, None) is not None:
                    setattr(data, k, None)
        dos = cfg["data"].get("dos_classes") or [c for c in cfg["data"]["classes"] if c != cfg["data"]["normal_class"]]
        self.dos_idx = [data.class_names.index(c) for c in dos if c in data.class_names]
        self.role = [c.role for c in data.clients]
        self.class_w = [self._class_weights(c.y_train) for c in data.clients]
        # Restricted softmax (FedRS-style): an honest drone only trains the outputs of classes it holds, so it
        # does not erase what other drones know about classes it has never seen. The poisoned drone is an
        # attacker running its own code, so it keeps plain cross-entropy.
        restrict = tc.get("local_loss", "ce") == "restricted_softmax"
        self.class_mask = [torch.as_tensor(np.bincount(c.y_train, minlength=self.C) > 0, device=self.device)
                           if restrict and c.role != "poisoned" else None for c in data.clients]

    def use_label_loss(self, kind):
        """Label-skew local losses, every drone runs the same code on its own training labels (server: FedAvg).
        fedrs - FedRS (Li & Zhan, KDD 2021; authors' code lxcnju/FedRepo algorithms/fedrs.py): logits of class c are
                multiplied by alpha + (1 - alpha) * n_c / max_c n_c, so classes a drone lacks are scaled by alpha
                (authors' runs: alpha 0.9 / 0.5 / 0.1).
        fedlc - FedLC (Zhang et al., ICML 2022, Eq. 7): logits calibrated as f_c - tau * n_c^(-1/4) before softmax
                cross-entropy; a class with no local samples gets n_c = 1e-8 (the term then dominates: it is
                effectively left out, which is what n_c^(-1/4) -> infinity means)."""
        self.label_loss = kind
        self.label_adj = []
        for c in self.data.clients:
            n = torch.as_tensor(np.bincount(c.y_train, minlength=self.C), dtype=torch.float32, device=self.device)
            if kind == "fedrs":
                a = float(self.cfg.get("fedrs", {}).get("alpha", 0.5))
                self.label_adj.append(a + (1 - a) * n / n.max())
            else:
                tau = float(self.cfg.get("fedlc", {}).get("tau", 1.0))
                self.label_adj.append(tau * n.clamp(min=1e-8).pow(-0.25))

    def use_fed_grab(self):
        """Fed-GraB (Xiao et al., NeurIPS 2023) local training for every drone. As in the authors' code
        (util/losses.py PIDLOSS): sigmoid (one-vs-rest) loss; per class, a PID controller (incremental form,
        Kp 10, Ki 0.01, Kd 0.1) drives the cumulative positive-minus-negative gradient towards 0 and maps its output
        to positive / negative gradient weights phi(u) = (10/9) / (1/9 + exp(-0.5 u)); each drone keeps its balancer
        state across rounds. Paper Eq. 4: the weights apply to class j only when r ~ U(0,1) > Pc[j], where Pc is the
        Direct Prior Analyzer's estimate = the global classifier's per-class weight norms scaled to sum to 1."""
        z = lambda: torch.zeros(self.C, device=self.device)
        self.grab = [{"pos": z(), "neg": z(), "out": z(), "e1": z(), "e2": z()} for _ in self.train_t]

    def _class_weights(self, y):
        """Local class weighting computed by each drone from its own labels only."""
        mode = self.cfg["training"]["class_weighting"]
        if mode == "none":
            return None
        counts = np.bincount(y, minlength=self.C).astype(np.float64)
        present = counts > 0
        w = np.zeros(self.C)
        ratio = counts[present].sum() / (present.sum() * counts[present])
        w[present] = np.sqrt(ratio) if mode == "sqrt_inverse" else ratio
        w[present] /= w[present].mean()
        return torch.as_tensor(w, dtype=torch.float32, device=self.device)

    def local_train(self, global_vec, cid, rnd):
        tc = self.cfg["training"]
        torch.manual_seed(self.seed * 1_000_003 + rnd * 1009 + cid)
        set_vector(self.model, global_vec)
        self.model.train()
        params = self.model.parameters()
        opt = torch.optim.Adam(params, lr=tc["lr"]) if tc["optimizer"] == "adam" \
            else torch.optim.SGD(params, lr=tc["lr"], momentum=tc.get("momentum", 0.0))
        X, y = self.train_t[cid]
        if getattr(self, "grab", None) is not None:
            return self._local_train_grab(cid, opt)
        mu = tc.get("prox_mu", 0.0)  # FedProx (Li et al., 2020): (mu/2)||w - w_global||^2 limits client drift
        anchor = global_vec.to(self.device) if mu > 0 else None
        bs, steps, done = tc["batch_size"], tc["local_steps"], 0
        while done < steps:
            perm = torch.randperm(len(y), device=self.device)
            for i in range(0, len(y), bs):
                if done == steps:
                    break
                b = perm[i:i + bs]
                logits = self.model(X[b].float())
                if self.class_mask[cid] is not None:
                    logits = logits.masked_fill(~self.class_mask[cid], -1e9)
                lk = getattr(self, "label_loss", None)
                if lk == "fedrs":
                    logits = logits * self.label_adj[cid]
                elif lk == "fedlc":
                    logits = logits - self.label_adj[cid]
                loss = F.cross_entropy(logits, y[b], weight=self.class_w[cid])
                if anchor is not None and self.role[cid] != "poisoned":
                    loss = loss + mu / 2 * (torch.nn.utils.parameters_to_vector(self.model.parameters()) - anchor).pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                done += 1
        return get_vector(self.model)

    def _local_train_grab(self, cid, opt):
        tc, st = self.cfg["training"], self.grab[cid]
        X, y = self.train_t[cid]
        with torch.no_grad():  # Direct Prior Analyzer on the received global classifier
            norms = self.model.net[-1].weight.norm(dim=1)
            prior = norms / norms.sum()
        # phi(u) = gamma / (1 + delta e^(-zeta u)) (paper); authors' code gamma 10, delta 9, zeta 0.5. delta = gamma - 1
        # keeps phi(0) = 1 (no re-weighting when balanced); gamma = the largest boost, left open by the paper.
        gamma = float(self.cfg.get("fed_grab", {}).get("gamma", 10.0))
        phi = lambda u: gamma / (1 + (gamma - 1) * torch.exp(-0.5 * u))
        # PID gains: default = the authors' code (Kp 10, Ki 0.01, Kd 0.1); the paper's Fig. 6 also tests
        # (1, 0, 0), (10, 0, 0) and (10, 0, 0.1). Chosen for Fed-GraB on its own attack-free validation score.
        pid = self.cfg.get("fed_grab", {})
        kp, ki, kd = pid.get("kp", 10.0), pid.get("ki", 0.01), pid.get("kd", 0.1)
        # gating: "eq4" = paper Eq. 4 with the DPA prior (default); "rank" = the authors' code (util/losses.py
        # hook_func_tensor): classes ranked head -> tail by the DPA prior, left unweighted with probability
        # 0.1^(rank/(C-1)) and then excluded from the gradient collector.
        rank_gate = None
        if pid.get("gating", "eq4") == "rank":
            rank = torch.empty(self.C, device=self.device)
            rank[torch.argsort(prior, descending=True)] = torch.arange(self.C, device=self.device, dtype=torch.float32)
            rank_gate = 0.1 ** (rank / (self.C - 1))
        bs, steps, done = tc["batch_size"], tc["local_steps"], 0
        while done < steps:
            perm = torch.randperm(len(y), device=self.device)
            for i in range(0, len(y), bs):
                if done == steps:
                    break
                b = perm[i:i + bs]
                logits = self.model(X[b].float())
                t = F.one_hot(y[b], self.C).float()
                # PID (incremental form, target 0) on the cumulative positive - negative gradient per class
                e0 = -(st["pos"] - st["neg"])
                st["out"] += kp * (e0 - st["e1"]) + ki * e0 + kd * (e0 - 2 * st["e1"] + st["e2"])
                st["e2"], st["e1"] = st["e1"], e0
                if rank_gate is None:
                    on = torch.rand(self.C, device=self.device) > prior  # Eq. 4
                else:  # authors' code: a class is left unweighted with prob 0.1^(rank/(C-1)), head (rank 0) always
                    on = torch.rand(self.C, device=self.device) >= rank_gate
                pos_w = torch.where(on, phi(st["out"]), torch.ones_like(prior))
                neg_w = torch.where(on, phi(-st["out"]), torch.ones_like(prior))
                w = (pos_w * t + neg_w * (1 - t)).detach()
                loss = (w * F.binary_cross_entropy_with_logits(logits, t, reduction="none")).sum() / len(b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():  # gradient collector: re-weighted |dL/dz| (Eq. 5)
                    g = (w * (torch.sigmoid(logits) - t)).abs() / len(b)
                    if rank_gate is not None:
                        g = g * on.float()  # authors' code: unweighted (gated-off) classes are not collected
                    st["pos"] += (g * t).sum(0)
                    st["neg"] += (g * (1 - t)).sum(0)
                done += 1
        return get_vector(self.model)

    @torch.no_grad()
    def predict(self, vec, X):
        set_vector(self.model, vec)
        self.model.eval()
        return torch.cat([self.model(X[i:i + 65536].float()).argmax(1) for i in range(0, len(X), 65536)]).cpu().numpy()

    def voter_score(self, vec, cid):
        return balanced_accuracy(self.vote_y[cid], self.predict(vec, self.vote_X[cid]), self.C)

    def voter_scores(self, vec, cids, kind="voter_score"):
        """One model scored on several drones' vote sets ({cid: score}); spread over worker processes when
        compute.parallel_drones is on (self.pool), otherwise one after another. Scores are pure functions of
        (model, vote set), so the order of computation does not change them."""
        pool = getattr(self, "pool", None)
        if pool is not None:
            return pool.scores(kind, vec, list(cids))
        return {c: getattr(self, kind)(vec, c) for c in cids}

    def voter_score_se(self, vec, cid):
        pred = self.predict(vec, self.vote_X[cid])
        y = self.vote_y[cid]
        cluster = self.cfg["jiang"].get("vote_se") == "cluster"
        groups = getattr(self.data.clients[cid], "vote_groups", None) if cluster else None
        return balanced_accuracy(y, pred, self.C), balanced_accuracy_se(y, pred, self.C, groups)

    def public_score(self, vec):
        """Server-side score on the public exam, used for the DQN state and reward."""
        X, y = self.public
        metric = self.cfg["jiang"]["dqn"].get("reward_metric", "balanced_accuracy")
        pred = self.predict(vec, X)
        if metric == "macro_f1_prior":
            return macro_f1_prior(y, pred, self.C, self.data.manifest["class_prior"])
        return (macro_f1 if metric == "macro_f1" else balanced_accuracy)(y, pred, self.C)

    def metrics(self, vec, split="val", confusion=False):
        X, y = {"val": self.val, "test": self.test, "test_pure": self.test_pure}[split]
        pred = self.predict(vec, X)
        C = self.C
        cm = np.zeros((C, C), dtype=np.int64)
        np.add.at(cm, (y, pred), 1)
        tp = np.diag(cm).astype(float)
        prec = np.divide(tp, cm.sum(0), out=np.zeros(C), where=cm.sum(0) > 0)
        rec = np.divide(tp, cm.sum(1), out=np.zeros(C), where=cm.sum(1) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(C), where=(prec + rec) > 0)
        attacks = y != self.data.normal_target
        names = self.data.class_names
        m = {
            "accuracy": float(tp.sum() / len(y)),
            "balanced_accuracy": float(rec.mean()),
            "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()), "macro_f1": float(f1.mean()),
            "rare_recall": float(rec[self.data.rare_target]), "rare_f1": float(f1[self.data.rare_target]),
            "missed_attack_rate": float((pred[attacks] == self.data.normal_target).mean()),
            # IDS view (attack vs normal, type ignored): false alarms = Normal flagged as any attack
            "false_alarm_rate": float((pred[~attacks] != self.data.normal_target).mean()) if (~attacks).any() else 0.0,
            "detection_rate": float((pred[attacks] != self.data.normal_target).mean()),
            "binary_precision": float((y[pred != self.data.normal_target] != self.data.normal_target).mean())
            if (pred != self.data.normal_target).any() else 0.0,
            "missed_by_class": {names[k]: float((pred[y == k] == self.data.normal_target).mean())
                                for k in range(C) if k != self.data.normal_target and (y == k).any()},
            # Main goal: DoS. Detection = DoS-family traffic flagged as any attack; type = macro F1 over the DoS types.
            "dos_detection_rate": float((pred[np.isin(y, self.dos_idx)] != self.data.normal_target).mean())
            if np.isin(y, self.dos_idx).any() else 0.0,
            "dos_type_macro_f1": float(f1[self.dos_idx].mean()),
            "per_class": {names[k]: {"precision": float(prec[k]), "recall": float(rec[k]), "f1": float(f1[k])}
                          for k in range(C)},
        }
        if split == "test" and getattr(self.data, "test_events", None) is not None:
            m.update(self.event_metrics(y, pred))
        if confusion:
            m["confusion"] = cm.tolist()
        return m

    def event_metrics(self, y, pred):
        """IDS view of the mixed-traffic test, per attack burst: was it detected, how many packets until the first
        alert, and how many false alarms per minute of normal traffic."""
        ev, nt = self.data.test_events, self.data.normal_target
        ep, t = ev["episode"], ev["time"]
        # Alert confirmation (standard IDS practice): raise an alert only when >= k of the last m packets are flagged.
        ac = self.cfg["data"].get("alert_confirm", {"k": 3, "m": 5})
        k, m = ac["k"], ac["m"]
        out = {}
        for tag in ("", "confirmed_"):
            detected, delays, fa, normal_time, fa_pk, n_norm, hit_pk, n_att = [], [], 0, 0.0, 0, 0, 0, 0
            for e in np.unique(ep):
                i = np.flatnonzero(ep == e)
                period = y[i] != nt
                alert = pred[i] != nt
                if tag:
                    c = np.cumsum(alert)
                    alert = (c - np.concatenate([np.zeros(m), c[:-m]])[:len(c)]) >= k
                if period.any():
                    first = np.flatnonzero(period)[0]
                    hits = np.flatnonzero(alert & period)
                    detected.append(len(hits) > 0)
                    if len(hits):
                        delays.append(int(hits[0] - first))
                fa += int((alert & ~period).sum())
                gaps = np.diff(t[i], prepend=t[i][0])
                normal_time += float(gaps[~period].sum())
                fa_pk += int((alert & ~period).sum()); n_norm += int((~period).sum())
                hit_pk += int((alert & period).sum()); n_att += int(period.sum())
            out.update({f"{tag}event_detection_rate": float(np.mean(detected)) if detected else 0.0,
                        f"{tag}event_delay_packets": float(np.median(delays)) if delays else float("nan"),
                        f"{tag}false_alarms_per_min": fa / max(normal_time / 60.0, 1e-9)})
            if tag:
                out.update(confirmed_false_alarm_rate=fa_pk / max(n_norm, 1),
                           confirmed_detection_rate=hit_pk / max(n_att, 1))
        return out
