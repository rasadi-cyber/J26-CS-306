"""Opt-in speed-up (compute.parallel_drones = N > 1; default off): within ONE job, the drones' local training and the
honest voters' scoring run in N CPU worker processes instead of one after another.

Results are meant to be bit-identical to the serial path:
- every drone's training is seeded per (seed, round, drone) inside FedSim.local_train, so where it runs does not matter;
- results are returned and combined in drone order (0..9), as in the serial loop;
- after a round, the parent's torch RNG is set to the state the serial loop leaves (that of the last drone trained);
- workers load the same partition from the shared memory-mapped copy (identical values, see data.save_shared).
Fed-GraB keeps a per-drone balancer state between rounds, so it always runs serially.
Useful where work is sequential (DQN pre-training chains, single-seed pilots); a full run already fills every core."""
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from .data import load_shared
from .sim import FedSim

_SIM = None  # this worker's replica of the job's simulator


def _init(cfg, seed, shared, method):
    global _SIM
    torch.set_num_threads(1)  # same arithmetic as the serial job (1 thread)
    data = load_shared(shared)
    if cfg.get("compute", {}).get("feature_dtype") == "float16":  # as runner.run_job (no-op on float16 copies)
        for c in data.clients:
            c.X_train, c.X_test = c.X_train.astype(np.float16, copy=False), c.X_test.astype(np.float16, copy=False)
        for k in ("X_public", "X_val", "X_test", "X_test_pure"):
            if getattr(data, k, None) is not None:
                setattr(data, k, getattr(data, k).astype(np.float16, copy=False))
    _SIM = FedSim(cfg, data, seed, None)
    if method.split("_")[0] in ("fedrs", "fedlc"):
        _SIM.use_label_loss(method.split("_")[0])


def _train(args):
    g, cid, rnd = args
    vec = _SIM.local_train(g, cid, rnd)
    return vec, torch.get_rng_state()


def _train_timed(args):
    t = time.perf_counter()
    vec, rng = _train(args)
    return vec, rng, (time.perf_counter() - t) * 1000


def _score(args):
    kind, vec, cid = args
    return getattr(_SIM, kind)(vec, cid)


class DronePool:
    def __init__(self, cfg, seed, shared, method, n):
        self.ex = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"), initializer=_init,
                                      initargs=(cfg, seed, shared, method))

    def train(self, g, cids, rnd, on_done=None):
        if on_done is None:
            res = list(self.ex.map(_train, [(g, c, rnd) for c in cids]))
            torch.set_rng_state(res[-1][1])  # leave the RNG where the serial loop would (after the last drone)
            return [vec for vec, _ in res]
        # on_done(cid, ms) is called as each drone finishes (secure-link live view); results still in drone order
        futs = {self.ex.submit(_train_timed, (g, c, rnd)): c for c in cids}
        done = {}
        for f in as_completed(futs):
            done[futs[f]] = f.result()
            on_done(futs[f], done[futs[f]][2])
        res = [done[c] for c in cids]
        torch.set_rng_state(res[-1][1])  # RNG state of the LAST drone, exactly as above
        return [vec for vec, _, _ in res]

    def scores(self, kind, vec, cids):
        return dict(zip(cids, self.ex.map(_score, [(kind, vec, c) for c in cids])))

    def close(self):
        self.ex.shutdown()
