"""Automatic goal check for a run: PASS / FAIL against the project's goals, so problems are caught right after a
run instead of by eye in the dashboard.

  .venv/bin/python check_goals.py RUN            # judged on validation (use for pilots / tuning)
  .venv/bin/python check_goals.py RUN --test     # judged on the test split (final runs only)
  .venv/bin/python check_goals.py RUN --ui       # also check that every dashboard tab renders
  .venv/bin/python check_goals.py RUN --test --compare v10-FINAL   # also flag anything worse than a reference run
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
KEYS = ("macro_f1", "accuracy", "dos_detection_rate", "dos_type_macro_f1", "false_alarm_rate", "missed_attack_rate",
        "rare_f1")


def load(run):
    meta = json.load(open(os.path.join(RUNS, run, "meta.json")))
    out = {}
    for m in meta["methods"]:
        per = []
        for s in meta["seeds"]:
            d = os.path.join(RUNS, run, m, f"seed_{s}")
            if not os.path.exists(os.path.join(d, "summary.json")):
                continue
            summ = json.load(open(os.path.join(d, "summary.json")))
            rounds = [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]
            per.append((summ, rounds))
        if per:
            out[m] = per
    return meta, out


def metric(per, k, test):
    if test:
        v = [s["test_last10"].get(k) for s, _ in per]
    else:
        v = [np.mean([r["val"][k] for r in rr[-10:] if r.get("val") and k in r["val"]]) for _, rr in per]
    v = [x for x in v if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def role(per, r, k):
    v = [s[r][k] for s, _ in per if s.get(r) and s[r].get(k) is not None]
    return float(np.mean(v)) if v else float("nan")


def main(argv):
    run, test, ui = argv[0], "--test" in argv, "--ui" in argv
    meta, res = load(run)
    split = "TEST" if test else "VALIDATION"
    checks = []
    add = lambda name, ok, detail: checks.append((name, bool(ok), detail))
    g = lambda m, k: metric(res[m], k, test) if m in res else float("nan")

    jiang = [m for m in res if m.startswith("jiang")]
    for m in jiang:
        add(f"{m}: poisoned drones never voted back in", role(res[m], "poisoned", "readmission_rate") <= 0.01,
            f"{role(res[m], 'poisoned', 'readmission_rate'):.0%}")
        add(f"{m}: rare drone voted back in >= 90%", role(res[m], "rare", "readmission_rate") >= 0.9,
            f"{role(res[m], 'rare', 'readmission_rate'):.0%}")
        lows = []
        for summ, rounds in res[m]:
            vs = summ.get("vote_start_round", 0)
            per = {}
            for r in rounds[vs:]:
                for d in r["clients"]:
                    if d["role"] == "general" and "readmitted" in d:
                        a = per.setdefault(d["cid"], [0, 0]); a[0] += d["readmitted"]; a[1] += 1
            lows += [b / n for b, n in per.values() if n >= 20]
        add(f"{m}: every regular drone voted back in >= 50% (when dropped)", min(lows, default=1) >= 0.5,
            f"lowest {min(lows, default=float('nan')):.0%}")
        add(f"{m}: rare share >= regular share", role(res[m], "rare", "mean_weight_share") >= role(res[m], "general", "mean_weight_share"),
            f"{role(res[m], 'rare', 'mean_weight_share'):.3f} vs {role(res[m], 'general', 'mean_weight_share'):.3f}")
    if "multi_krum" in res:
        add("multi_krum: cuts the rare drone (share ~0)", role(res["multi_krum"], "rare", "mean_weight_share") < 0.01,
            f"{role(res['multi_krum'], 'rare', 'mean_weight_share'):.3f}")
        add("multi_krum: cuts the poisoned drones (share ~0)", role(res["multi_krum"], "poisoned", "mean_weight_share") < 0.01,
            f"{role(res['multi_krum'], 'poisoned', 'mean_weight_share'):.3f}")
    # The two literature poles the proposal compares against (expected trend, not tuned for):
    if "fedwgca" in res:  # robustness side: suppresses the rare drone (cut or down-weighted below a regular drone)
        add("fedwgca (robust side): rare drone cut or down-weighted (share < regular)",
            role(res["fedwgca"], "rare", "mean_weight_share") < role(res["fedwgca"], "general", "mean_weight_share"),
            f"rare {role(res['fedwgca'], 'rare', 'mean_weight_share'):.3f} vs regular {role(res['fedwgca'], 'general', 'mean_weight_share'):.3f}")
        add("fedwgca (robust side): poisoned drones cut or down-weighted (share < regular)",
            role(res["fedwgca"], "poisoned", "mean_weight_share") < role(res["fedwgca"], "general", "mean_weight_share"),
            f"poisoned {role(res['fedwgca'], 'poisoned', 'mean_weight_share'):.3f}")
    if "fed_grab" in res:  # rare side: keeps the rare class but lets the poisoned drones in
        add("fed_grab (rare side): poisoned drones let in (share > 0)", role(res["fed_grab"], "poisoned", "mean_weight_share") > 0.05,
            f"{role(res['fed_grab'], 'poisoned', 'mean_weight_share'):.3f}")
    best = next((m for m in ("jiang_dqn_pre", "jiang_dqn", "jiang_fixed") if m in res), None)
    for base in ("fedwgca", "fed_grab"):
        if best and base in res:
            add(f"{best} macro F1 > {base}", g(best, "macro_f1") > g(base, "macro_f1"), f"{g(best, 'macro_f1'):.3f} vs {g(base, 'macro_f1'):.3f}")
            add(f"{best} misses fewer attacks than {base}", g(best, "missed_attack_rate") <= g(base, "missed_attack_rate"),
                f"{g(best, 'missed_attack_rate'):.1%} vs {g(base, 'missed_attack_rate'):.1%}")
    if best and "fedavg" in res:
        add(f"{best} misses fewer attacks than FedAvg", g(best, "missed_attack_rate") < g("fedavg", "missed_attack_rate"),
            f"{g(best, 'missed_attack_rate'):.1%} vs {g('fedavg', 'missed_attack_rate'):.1%}")
    if best and "multi_krum" in res:
        add(f"{best} macro F1 > Multi-Krum", g(best, "macro_f1") > g("multi_krum", "macro_f1"),
            f"{g(best, 'macro_f1'):.3f} vs {g('multi_krum', 'macro_f1'):.3f}")
    if "jiang_fixed" in res and "jiang_dqn" in res:
        add("online DQN >= fixed weight (macro F1)", g("jiang_dqn", "macro_f1") >= g("jiang_fixed", "macro_f1") - 0.003,
            f"{g('jiang_dqn', 'macro_f1'):.3f} vs {g('jiang_fixed', 'macro_f1'):.3f}")
    if "jiang_dqn_pre" in res and "jiang_dqn" in res:
        add("pre-trained DQN >= online DQN (macro F1)", g("jiang_dqn_pre", "macro_f1") >= g("jiang_dqn", "macro_f1") - 0.003,
            f"{g('jiang_dqn_pre', 'macro_f1'):.3f} vs {g('jiang_dqn', 'macro_f1'):.3f}")
    if best:
        for k, lo in (("accuracy", 0.80), ("dos_detection_rate", 0.80), ("macro_f1", 0.80)):
            add(f"{best}: {k} >= {lo}", g(best, k) >= lo, f"{g(best, k):.3f}")
        add(f"{best}: false alarms <= 5%", g(best, "false_alarm_rate") <= 0.05, f"{g(best, 'false_alarm_rate'):.1%}")
        if test and not np.isnan(g(best, "confirmed_false_alarm_rate")):
            add(f"{best}: false alarms with alert confirmation <= 2%", g(best, "confirmed_false_alarm_rate") <= 0.02,
                f"{g(best, 'confirmed_false_alarm_rate'):.1%} (raw {g(best, 'false_alarm_rate'):.1%})")
        if test:
            cm = sum(np.array(s.get("confusion_last10", s["test"]["confusion"])) for s, _ in res[best])
            diag = np.diag(cm) / cm.sum(1).clip(min=1)
            n_perfect = int((diag >= 0.995).sum())
            add(f"{best}: confusion matrix not suspiciously perfect (<= 2 rows at >= 99.5%)", n_perfect <= 2,
                f"{n_perfect} rows at >= 99.5%: " + ", ".join(f"{d:.1%}" for d in diag))
    # --- watch for NEW problems, not just known ones: per seed, per drone, per class ---
    for m in jiang:
        for summ, rounds in res[m]:
            sd = summ["seed"]
            p_in = (summ.get("poisoned") or {}).get("readmission_rate")
            r_in = (summ.get("rare") or {}).get("readmission_rate")
            if p_in is not None and p_in > 0.01:
                add(f"{m} seed {sd}: poisoned voted back in", False, f"{p_in:.0%}")
            if r_in is not None and r_in < 0.9:
                add(f"{m} seed {sd}: rare drone voted back in < 90%", False, f"{r_in:.0%}")
            vs = summ.get("vote_start_round", 0)
            per = {}
            for r in rounds[vs:]:
                for d in r["clients"]:
                    if d["role"] in ("general", "rare") and "readmitted" in d:
                        a = per.setdefault((d["cid"], d["role"]), [0, 0]); a[0] += d["readmitted"]; a[1] += 1
            for (cid, rl), (b, n) in per.items():
                if n >= 20 and b / n < 0.5:
                    add(f"{m} seed {sd}: honest drone {cid + 1} ({rl}) voted back in < 50%", False, f"{b / n:.0%} of {n}")
    if "multi_krum" in res:
        for summ, _ in res["multi_krum"]:
            ps = (summ.get("poisoned") or {}).get("mean_weight_share", 0)
            if ps >= 0.01:
                add(f"multi_krum seed {summ['seed']}: keeps poisoned drones", False, f"share {ps:.3f}")
    if best and test:
        pc = {}
        for summ, _ in res[best]:
            for cl, v in summ["test"]["per_class"].items():
                pc.setdefault(cl, []).append(v["f1"])
        weak = {cl: np.mean(v) for cl, v in pc.items() if np.mean(v) < 0.3}
        add(f"{best}: no class collapsed (F1 >= 0.3)", not weak, ", ".join(f"{k} {v:.2f}" for k, v in weak.items()) or "ok")
    ref = next((argv[i + 1] for i, a in enumerate(argv) if a == "--compare" and i + 1 < len(argv)), None)
    if ref:
        _, rres = load(ref)
        for m in [m for m in res if m in rres]:
            for k, worse_if_up, tol in (("macro_f1", False, 0.02), ("accuracy", False, 0.02), ("dos_detection_rate", False, 0.01),
                                        ("dos_type_macro_f1", False, 0.02), ("rare_f1", False, 0.03),
                                        ("missed_attack_rate", True, 0.01), ("false_alarm_rate", True, 0.01)):
                a, b = metric(res[m], k, test), metric(rres[m], k, test)
                if np.isnan(a) or np.isnan(b):
                    continue
                worse = (a - b > tol) if worse_if_up else (b - a > tol)
                if worse:
                    add(f"REGRESSION vs {ref}: {m} {k}", False, f"{a:.3f} now vs {b:.3f}")
    if ui:
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300)
        at.session_state["run"] = run
        at.run()
        add("dashboard renders every tab without errors", not at.exception, str([e.value for e in at.exception])[:200])

    print(f"\nGoal check for {run} ({split}):")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  [{detail}]")
    print(f"  -> {sum(ok for _, ok, _ in checks)}/{len(checks)} passed")


if __name__ == "__main__":
    main(sys.argv[1:])
