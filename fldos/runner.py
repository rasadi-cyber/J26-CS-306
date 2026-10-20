"""Runs every (method, seed) job, in parallel processes, and writes the logs the dashboard reads.

  <run>/meta.json                          status, config, seeds, roles and partition per seed
  <run>/<method>/seed_<s>/rounds.jsonl     per round: validation metrics + every drone's decision
  <run>/<method>/seed_<s>/summary.json     final test metrics, confusion matrix, rare / poisoned handling
  <run>/<method>/seed_<s>/dqn.pt           trained DQN (DQN methods)
  <run>/results.json                       mean / std across seeds per method
"""
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch

from . import mtls, securelink
from .aggregators import Update, build
from .data import make_partition, partition_cached, shared_dir, save_shared, load_shared
from .dqn import DQNAgent
from .sim import FedSim


LAST10_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "rare_recall", "rare_f1", "missed_attack_rate",
               "false_alarm_rate", "detection_rate", "binary_precision", "dos_detection_rate", "dos_type_macro_f1",
               "event_detection_rate", "event_delay_packets", "false_alarms_per_min",
               "confirmed_event_detection_rate", "confirmed_event_delay_packets", "confirmed_false_alarms_per_min",
               "confirmed_false_alarm_rate", "confirmed_detection_rate")


def write_json(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def role_stats(rounds, role, vote_start):
    rows = [d for r in rounds[vote_start:] for d in r["clients"] if d["role"] == role]
    if not rows:
        return None
    voted = [d for d in rows if "votes_yes" in d]
    readm = [d for d in voted if d["readmitted"]]
    return {
        "filtered_out_rate": float(np.mean([d.get("filtered_out", False) for d in rows])),
        "votes_held": len(voted),
        "readmitted": len(readm),
        "readmission_rate": float(len(readm) / len(voted)) if voted else None,
        "mean_yes_votes": float(np.mean([d["votes_yes"] for d in voted])) if voted else None,
        "inclusion_rate": float(np.mean([d["weight"] > 0 for d in rows])),
        "mean_weight_share": float(np.mean([d["weight"] for d in rows])),
        "mean_dqn_weight": float(np.mean([d["dqn_weight"] for d in readm if d.get("dqn_weight") is not None]))
        if any(d.get("dqn_weight") is not None for d in readm) else None,
    }


def poison_update(cfg, role, g, u):
    """Model-poisoning on top of (or instead of) label flipping, applied to what a poisoned drone sends.
      scale      boosted update g + f (u - g)      (Bhagoji et al., 2019; f = 10)
      sign_flip  reversed update g - f (u - g)     (sign-flipping attack; f = 1)"""
    mode = cfg["clients"].get("model_attack", "none")
    if role != "poisoned" or mode == "none":
        return u
    f = cfg["clients"].get("model_attack_factor", 10.0 if mode == "scale" else 1.0)
    return g + f * (u - g) if mode == "scale" else g - f * (u - g)


def _timed_train(sim, g, cid, rnd, chan):
    t = time.perf_counter()
    vec = sim.local_train(g, cid, rnd)
    chan.trained(cid, (time.perf_counter() - t) * 1000)
    return vec


def run_job(cfg, root, run_dir, method, seed, agent_state=None, log=print, device=None):
    torch.set_num_threads(1)
    out = os.path.join(run_dir, method, f"seed_{seed}")
    os.makedirs(out, exist_ok=True)
    if cfg.get("compute", {}).get("shared_data"):  # one memory-mapped copy per seed, shared by all its jobs
        data = load_shared(shared_dir(cfg, seed, root))
    else:
        data = make_partition(cfg, seed, root, log, save_files=False)
    if cfg.get("compute", {}).get("feature_dtype") == "float16":  # half-precision storage (RAM); compute stays float32
        # copy=False: shared (already float16, memory-mapped) arrays are used as they are; values are identical
        for c in data.clients:
            c.X_train, c.X_test = c.X_train.astype(np.float16, copy=False), c.X_test.astype(np.float16, copy=False)
        for k in ("X_public", "X_val", "X_test", "X_test_pure"):
            if getattr(data, k, None) is not None:
                setattr(data, k, getattr(data, k).astype(np.float16, copy=False))
    sim = FedSim(cfg, data, seed, device)
    if method == "fed_grab":
        sim.use_fed_grab()
    if method.split("_")[0] in ("fedrs", "fedlc"):  # label-skew local losses (server: FedAvg)
        sim.use_label_loss(method.split("_")[0])
    # Resume (training.resume_from = an earlier run folder with the same settings): continue each job from its saved
    # global model, round log, DQN and Fed-GraB balancer state, so extending training costs only the extra rounds.
    res_src = cfg["training"].get("resume_from")
    res_src = os.path.join(root, res_src, method, f"seed_{seed}") if res_src else None
    agent = None
    if method.startswith("jiang_dqn"):
        agent = DQNAgent(cfg["jiang"]["dqn"], seed)
        if res_src:
            agent.load_state_dict(torch.load(os.path.join(res_src, "dqn.pt"), weights_only=False))
            if method in ("jiang_dqn_pre", "jiang_dqn_prexv"):
                agent.eps_override = cfg["jiang"]["dqn"]["pretrained_eps"]
                agent.frozen = cfg["jiang"]["dqn"].get("pretrained_frozen", False)
        elif method in ("jiang_dqn_pre", "jiang_dqn_prexv"):
            from_file = agent_state is None
            if from_file:
                path = dqn_store_path(cfg, root, "xv" if method == "jiang_dqn_prexv" else "setup")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} missing: run `run_experiment.py --pretrain-dqn` first")
                agent_state = torch.load(path, weights_only=False)
            agent.load_state_dict(agent_state)
            agent.eps_override = cfg["jiang"]["dqn"]["pretrained_eps"]
            # Frozen only when used for evaluation (loaded from file), never while the pre-training chain runs.
            agent.frozen = from_file and cfg["jiang"]["dqn"].get("pretrained_frozen", False)
    rule = build(method, cfg, sim, seed, agent)
    # compute.parallel_drones = N > 1 (opt-in, default off): drones train and honest voters score in N worker
    # processes; bit-identical to the serial path (see fldos/parallel.py). CPU only; Fed-GraB stays serial.
    pool = None
    par = int(cfg.get("compute", {}).get("parallel_drones", 0) or 0)
    # security.transport = "mtls" (opt-in, default "inproc"): every drone is its own process, connected to this job's
    # gRPC server over TLS 1.3 with mutual X.509 authentication (fldos/mtls.py); same DronePool interface and identity
    # rules, so results are bit-identical. Fed-GraB keeps per-drone balancer state between rounds: serial in-process.
    chan = None
    if securelink.enabled(cfg):
        sc = cfg["security"]
        chan = securelink.SecureChannel([c.cid for c in data.clients], out, sc.get("bandwidth_mbps", 10.0),
                                        sc.get("latency_ms", 20.0), log, tamper_prob=sc.get("tamper_prob", 0.0),
                                        max_retries=sc.get("max_retries", 3), seed=seed)
    if mtls.enabled(cfg) and method == "fed_grab":
        log(f"[mtls] {method}: Fed-GraB keeps per-drone balancer state, so it runs serially in-process (no mTLS)")
    if mtls.enabled(cfg) and method != "fed_grab" and device in (None, "cpu"):
        sd = shared_dir(cfg, seed, root)
        if not os.path.exists(sd):
            save_shared(data, sd, cfg.get("compute", {}).get("feature_dtype") == "float16")
        pool = sim.pool = mtls.MTLSPool(cfg, seed, sd, method, [c.cid for c in data.clients],
                                        mtls.pki_dir_for(cfg, root, run_dir), out, log,
                                        own_live=not securelink.enabled(cfg),
                                        material=chan.drone_material if chan is not None else None)
    elif par > 1 and method != "fed_grab" and device in (None, "cpu"):
        from .parallel import DronePool
        sd = shared_dir(cfg, seed, root)
        if not os.path.exists(sd):
            save_shared(data, sd, cfg.get("compute", {}).get("feature_dtype") == "float16")
        pool = sim.pool = DronePool(cfg, seed, sd, method, par)
    # security.secure_link (opt-in, default off): every model crosses a simulated encrypted, signed link (drone ->
    # server, server -> drones, server -> voters and their scores back); decrypted copies are byte-identical, so
    # results are unchanged. security.tamper_prob (default 0) simulates interceptions with its own RNG; rejected
    # packets are NACKed and re-sent (security.max_retries, default 3), so the round carries on.
    remote = isinstance(pool, mtls.MTLSPool) and chan is not None  # drone processes seal / open their own messages
    if chan is not None:
        sim.voter_scores = chan.remote_voter_scores(pool) if remote else chan.wrap_voter_scores(sim.voter_scores)

    R = cfg["training"]["rounds"]
    vote_start = int(round(cfg["jiang"]["vote_start_frac"] * R))
    g = sim.init_vec.clone()
    rounds = []
    path = os.path.join(out, "rounds.jsonl")
    if res_src:
        g = torch.load(os.path.join(res_src, "global_model.pt"))
        rounds = [json.loads(l) for l in open(os.path.join(res_src, "rounds.jsonl"))]
        gp = os.path.join(res_src, "grab_state.pt")
        if method == "fed_grab" and os.path.exists(gp):
            sim.grab = torch.load(gp)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rounds)
    start = len(rounds)
    if res_src:  # keep the original vote start: voting already began in the earlier part of the run
        vote_start = json.load(open(os.path.join(res_src, "summary.json"))).get("vote_start_round", vote_start)
    if remote:  # the starting global model handed out to every drone process (sealed, opened by each drone)
        chan.downlink_remote(start - 1, g, pool.deliver, record=False)
    t_start = time.time()
    last = []  # test metrics of the last 10 global models (less noisy than one final snapshot)
    last_pure = []  # mixed-traffic setups: the same models on the pure per-capture test (clean reference)
    try:
        for rnd in range(start, R):
            t0 = time.time()
            if chan is None:
                vecs = pool.train(g, [c.cid for c in data.clients], rnd) if pool is not None \
                    else [sim.local_train(g, c.cid, rnd) for c in data.clients]
            elif remote:  # real drone processes: each trains from the global it received and seals its own update
                chan.begin_round(rnd)
                sealed = pool.train_sealed([c.cid for c in data.clients], rnd, on_done=chan.trained)
            else:  # same training, timed per drone; each drone is marked ready (live view) as soon as it finishes
                chan.begin_round(rnd)
                vecs = pool.train(g, [c.cid for c in data.clients], rnd, on_done=chan.trained) if pool is not None \
                    else [_timed_train(sim, g, c.cid, rnd, chan) for c in data.clients]
            if remote:  # the server opens + verifies each drone's packet (the drone applied its own behaviour)
                chan.phase("uplink")
                updates = [Update(c.cid, c.role, chan.uplink_remote(rnd, c.cid, sealed[c.cid][0], sealed[c.cid][1],
                                                                    pool.reseal, g, g, sealed[c.cid][2]), len(c.y_train))
                           for c in data.clients]
                chan.phase("server")
            else:
                updates = [Update(c.cid, c.role, poison_update(cfg, c.role, g, v), len(c.y_train))
                           for c, v in zip(data.clients, vecs)]
            if chan is not None and not remote:  # each drone's update over its link; the server uses the decrypted copy
                chan.phase("uplink")
                updates = [Update(u.cid, u.role, chan.uplink(rnd, u.cid, u.vec, g), u.n) for u in updates]
                chan.phase("server")
            g_prev = g
            g, rows, info = rule.aggregate(rnd, updates, g)
            if chan is not None:  # the new global model goes back to every drone
                chan.downlink_remote(rnd, g, pool.deliver) if remote else chan.downlink(rnd, g)
                chan.end_round()
            if rnd in set(cfg["training"].get("snapshot_rounds", [])):  # opt-in: model weights for map figures
                os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)
                torch.save({"round": rnd, "global_before": g_prev.detach().cpu().half(), "global_after": g.detach().cpu().half(),
                            "locals": torch.stack([u.vec.detach().cpu() for u in updates]).half(),
                            "cid": [u.cid for u in updates], "role": [u.role for u in updates], "rows": rows},
                           os.path.join(out, "snapshots", f"round_{rnd:04d}.pt"))
            if rnd >= R - 10:
                last.append(sim.metrics(g, "test", confusion=True))
                if sim.test_pure is not None:
                    last_pure.append(sim.metrics(g, "test_pure", confusion=True))
            every = cfg["training"].get("eval_every", 1)  # validation scoring cadence (always the last 10 rounds)
            val = sim.metrics(g, "val") if (rnd % every == 0 or rnd >= R - 10) else None
            rec = {"round": rnd, "val": val, "clients": rows, "info": info,
                   "time_s": round(time.time() - t0, 3)}
            rounds.append(rec)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if ((rnd + 1) % max(1, R // 10) == 0 or rnd == R - 1) and rec["val"] is not None:
                v = rec["val"]
                tags = " ".join(f"{d['role'][0].upper()}{d['cid'] + 1}:{d['status'][:4]}" for d in rows if d["role"] != "general")
                log(f"[{method} s{seed}] {rnd + 1:4d}/{R} macroF1={v['macro_f1']:.3f} rare-F1={v['rare_f1']:.3f} "
                    f"missed-attacks={v['missed_attack_rate']:.3f}  {tags}")
    except BaseException:  # stop the drone processes / mTLS server at once (a dead job leaves no drones behind)
        if pool is not None:
            getattr(pool, "abort", pool.close)()
        raise

    if pool is not None:
        pool.close()
        sim.pool = None
    if chan is not None:
        chan.close()
    summary = {"method": method, "seed": seed, "class_names": data.class_names, "rounds": R,
               "vote_start_round": vote_start, "minutes": round((time.time() - t_start) / 60, 2),
               "test": sim.metrics(g, "test", confusion=True), "val_final": rounds[-1]["val"],
               "confusion_last10": np.sum([m["confusion"] for m in last], axis=0).tolist(),
               "test_last10": {k: float(np.mean([m[k] for m in last])) for k in LAST10_KEYS if k in last[0]},
               "rare": role_stats(rounds, "rare", vote_start), "poisoned": role_stats(rounds, "poisoned", vote_start),
               "general": role_stats(rounds, "general", vote_start)}
    if last_pure:
        summary["test_pure_last10"] = {k: float(np.mean([m[k] for m in last_pure])) for k in LAST10_KEYS if k in last_pure[0]}
        summary["confusion_pure_last10"] = np.sum([m["confusion"] for m in last_pure], axis=0).tolist()
    write_json(os.path.join(out, "summary.json"), summary)
    torch.save(g.cpu(), os.path.join(out, "global_model.pt"))
    if getattr(sim, "grab", None) is not None:  # Fed-GraB balancer state per drone (needed to resume exactly)
        torch.save(sim.grab, os.path.join(out, "grab_state.pt"))
    if agent is not None:
        torch.save(agent.state_dict(), os.path.join(out, "dqn.pt"))
    return summary, (agent.state_dict() if agent is not None else None)


def _job(args):
    cfg, root, run_dir, method, seed, device = args
    try:
        summary, _ = run_job(cfg, root, run_dir, method, seed, log=lambda m: print(m, flush=True), device=device)
        return method, seed, summary, None
    except Exception as e:  # report, keep the other jobs running
        return method, seed, None, f"{e}\n{traceback.format_exc()}"


def aggregate_results(run_dir, methods, seeds):
    keys = list(LAST10_KEYS)
    res = {}
    for m in methods:
        sums = []
        for s in seeds:
            p = os.path.join(run_dir, m, f"seed_{s}", "summary.json")
            if os.path.exists(p):
                with open(p) as f:
                    sums.append(json.load(f))
        if not sums:
            continue
        entry = {"seeds": [x["seed"] for x in sums]}
        for k in keys:
            vals = [x["test"][k] for x in sums if k in x["test"]]
            if not vals:  # metric added after this run was made
                continue
            entry[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            vals = [x["test_last10"][k] for x in sums if k in x["test_last10"]]
            if not vals:
                continue
            entry[f"{k}_last10"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for role in ("rare", "poisoned"):
            for k in ("inclusion_rate", "readmission_rate", "mean_weight_share", "filtered_out_rate"):
                vals = [x[role][k] for x in sums if x.get(role) and x[role].get(k) is not None]
                if vals:
                    entry[f"{role}_{k}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        res[m] = entry
    write_json(os.path.join(run_dir, "results.json"), res)
    return res


def run(cfg, root, run_dir, workers=None, log=print):
    os.makedirs(run_dir, exist_ok=True)
    meta_path = os.path.join(run_dir, "meta.json")
    seeds, methods = cfg["seeds"], cfg["methods"]
    meta = {"status": "preparing", "started": time.time(), "config": cfg, "methods": methods, "seeds": seeds,
            "rounds": cfg["training"]["rounds"], "pid": os.getpid(), "errors": []}
    write_json(meta_path, meta)
    try:
        partitions = {}
        for s in seeds:  # build + save each seed's partition once, before the workers start
            # cached partition: reuse it (identical rows); otherwise build it once and save the per-drone files
            d = make_partition(cfg, s, root, log, save_files=not partition_cached(cfg, s, root))
            partitions[str(s)] = {"roles": {c.cid: c.role for c in d.clients},
                                  "clients": [{"cid": c.cid, "role": c.role, "n_train": len(c.y_train),
                                               "n_vote_set": len(c.y_test), "counts": c.counts} for c in d.clients],
                                  "n_val": len(d.y_val), "n_test": len(d.y_test), "n_public": len(d.y_public)}
            meta.update(class_names=d.class_names, rare_class=cfg["data"]["rare_class"])
            if cfg.get("compute", {}).get("shared_data") and not os.path.exists(shared_dir(cfg, s, root)):
                save_shared(d, shared_dir(cfg, s, root), cfg.get("compute", {}).get("feature_dtype") == "float16")
                log(f"seed {s}: shared memory-mapped copy written")
        meta.update(status="running", partitions=partitions,
                    vote_start_round=int(round(cfg["jiang"]["vote_start_frac"] * cfg["training"]["rounds"])))
        write_json(meta_path, meta)

        # compute.gpu_workers > 0: some seeds run every method on the GPU (MPS) next to the CPU pool. Whole seeds
        # are assigned, so every method gets the same CPU/GPU mix. Outside "training", so DQN setup keys are unchanged.
        gw = int(cfg.get("compute", {}).get("gpu_workers", 0))
        workers = workers or max(1, min(len(seeds) * len(methods), (os.cpu_count() or 2) - 2))
        n_gpu_seeds = round(len(seeds) * gw / (gw + workers)) if gw else 0
        gpu_seeds = set(seeds[len(seeds) - n_gpu_seeds:]) if n_gpu_seeds else set()
        jobs = [(cfg, root, run_dir, m, s, "mps" if s in gpu_seeds else None) for s in seeds for m in methods]
        meta["gpu_seeds"] = sorted(gpu_seeds)
        if securelink.enabled(cfg):  # keys are enrolled per job: each job's fingerprints are added as it finishes
            sc = cfg["security"]
            meta["secure_link"] = {**securelink.ALGOS, "bandwidth_mbps": sc.get("bandwidth_mbps", 10.0),
                                   "latency_ms": sc.get("latency_ms", 20.0), "tamper_prob": sc.get("tamper_prob", 0.0),
                                   "max_retries": sc.get("max_retries", 3), "enrollment": [], "jobs": {}}
            write_json(meta_path, meta)
        log(f"{len(jobs)} jobs on {workers} CPU worker processes" + (f" + {gw} GPU workers (seeds {sorted(gpu_seeds)})" if gw else ""))
        pools = [ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))]
        if gpu_seeds:
            pools.append(ProcessPoolExecutor(max_workers=gw, mp_context=mp.get_context("spawn")))
        with pools[0] as ex, (pools[1] if len(pools) > 1 else pools[0]) as gx:
            futs = [(gx if j[5] else ex).submit(_job, j) for j in jobs]
            for fut in as_completed(futs):
                m, s, summary, err = fut.result()
                if err:
                    meta["errors"].append({"method": m, "seed": s, "error": err})
                    log(f"!! {m} seed {s} failed:\n{err}")
                else:
                    if "secure_link" in meta:
                        p = os.path.join(run_dir, m, f"seed_{s}", "securelink_meta.json")
                        if os.path.exists(p):
                            sl = json.load(open(p))
                            meta["secure_link"]["jobs"][f"{m}/seed_{s}"] = sl
                            meta["secure_link"]["enrollment"] = meta["secure_link"]["enrollment"] or sl["enrollment"]
                    t = summary["test"]
                    log(f"done {m} seed {s}: test macroF1={t['macro_f1']:.4f} rare-F1={t['rare_f1']:.4f}")
                write_json(meta_path, meta)
        aggregate_results(run_dir, methods, seeds)
        meta.update(status="failed" if meta["errors"] else "done", finished=time.time())
    except Exception as e:
        meta.update(status="failed", finished=time.time(), errors=meta["errors"] + [{"error": f"{e}\n{traceback.format_exc()}"}])
        raise
    finally:
        write_json(meta_path, meta)


def setup_key(cfg):
    """Fingerprint of everything that defines the DQN's environment (data, swarm, training, Jiang settings)."""
    keep = {k: cfg[k] for k in ("data", "clients", "training", "multi_krum") if k in cfg}
    j = json.loads(json.dumps(cfg["jiang"]))
    for k in ("pretrained_path", "pretrained_eps", "pretrained_frozen"):
        j.get("dqn", {}).pop(k, None)
    keep["jiang"] = j
    import hashlib
    return hashlib.md5(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def dqn_store_path(cfg, root, mode):
    """setup: one DQN per setup, accumulated over every pre-training (non-final) run of that setup.
    xv:    one DQN carried across versions (setups); its network is kept when the setup changes."""
    name = f"pre_setup_{setup_key(cfg)}.pt" if mode == "setup" else "pre_xv.pt"
    # store_dir keeps a dataset's DQNs apart (e.g. ISOT uses dqn_store/isot); default = the UAV-NIDD store
    return os.path.join(root, cfg["jiang"]["dqn"].get("store_dir", "dqn_store"), name)


def pretrain_dqn(cfg, root, seeds, out_path=None, log=print, mode="setup", init_from=None):
    """Continue a stored DQN through sequential runs on non-final seeds and save it back (it keeps accumulating
    across invocations). mode=setup: per setup. mode=xv: across versions - when the stored DQN comes from another
    setup, its network is kept and its replay memory / input statistics are reset (transfer, then fine-tune)."""
    if any(int(x) < 100 for x in seeds):
        raise ValueError("pre-training uses non-final seeds only (>= 100); seeds 1-5 are the final evaluation")
    path = out_path or dqn_store_path(cfg, root, mode)
    lineage_path = path.replace(".pt", "_lineage.json")
    lineage = json.load(open(lineage_path)) if os.path.exists(lineage_path) else []
    key = setup_key(cfg)
    state = torch.load(path, weights_only=False) if os.path.exists(path) else None
    if state is None and mode == "xv" and init_from and os.path.exists(init_from):
        state = torch.load(init_from, weights_only=False)
        state["setup_key"] = "previous-version"
        lineage.append({"event": "initialised from previous version", "file": init_from,
                        "decisions": state.get("decisions"), "time": time.strftime("%Y-%m-%d %H:%M")})
    if state is not None and mode == "xv" and state.get("setup_key") != key:
        tmp = DQNAgent(cfg["jiang"]["dqn"], 0)
        tmp.load_state_dict(state)
        tmp.reset_for_new_setup()
        prev = state.get("setup_key")
        state = tmp.state_dict()
        lineage.append({"event": "new setup: network kept, replay memory and input statistics reset",
                        "from_setup": prev, "to_setup": key, "time": time.strftime("%Y-%m-%d %H:%M")})
    tag = os.path.basename(os.path.normpath(cfg["data"].get("processed_dir", "data")))  # one folder per dataset
    run_dir = os.path.join(root, "runs", f"dqn-pretrain-{mode}-{tag}-{time.strftime('%Y%m%d-%H%M%S')}")
    method = "jiang_dqn_prexv" if mode == "xv" else "jiang_dqn_pre"
    for s in seeds:
        log(f"pre-training DQN ({mode}) on non-final seed {s}")
        # threads for the sequential pre-training chain (env PRETRAIN_THREADS when it shares the machine with runs)
        torch.set_num_threads(int(os.environ.get("PRETRAIN_THREADS", max(1, (os.cpu_count() or 2) - 2))))
        # compute.pretrain_eval_every (opt-in): validation scoring cadence during pre-training. Validation only feeds
        # the round log (no curves are shown for pre-training), never training or the DQN, so the DQN is unchanged.
        ev = cfg.get("compute", {}).get("pretrain_eval_every")
        cfg_run = json.loads(json.dumps(cfg))
        if ev:
            cfg_run["training"]["eval_every"] = int(ev)
        _, state = run_job(cfg_run, root, run_dir, method if state is not None else "jiang_dqn", s, agent_state=state, log=log)
        lineage.append({"event": "trained", "setup": key, "seed": s, "decisions": state["decisions"],
                        "time": time.strftime("%Y-%m-%d %H:%M")})
    state["setup_key"] = key
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)
    log(f"saved {mode} DQN -> {path} ({state['decisions']} decisions in total)")
