"""Markdown results table for a run (TEST, last 10 models, mean over seeds), optionally next to an older run.
   python tools/results_table.py NEW_RUN [OLD_RUN]"""
import json, sys

new = sys.argv[1]; old = sys.argv[2] if len(sys.argv) > 2 else None
N = json.load(open(f"runs/{new}/results.json")); O = json.load(open(f"runs/{old}/results.json")) if old else {}
rare = json.load(open(f"runs/{new}/meta.json")).get("rare_class", "rare")
LAB = {"fedavg_clean": "FedAvg, no attackers (reference)", "fedavg": "FedAvg", "multi_krum": "Multi-Krum",
       "fedwgca": "FedWGCA (robust side)", "fed_grab": "Fed-GraB (rare side)", "jiang_fixed": "Jiang, fixed weight",
       "jiang_dqn": "Jiang + DQN (online)", "jiang_dqn_pre": "Jiang + DQN pre-trained",
       "jiang_dqn_prexv": "Jiang + DQN across versions"}
K = [("macro_f1_last10", "Macro F1", "{:.3f}"), ("accuracy_last10", "Accuracy", "{:.3f}"),
     ("rare_recall_last10", f"{rare} recall", "{:.3f}"), ("rare_f1_last10", f"{rare} F1", "{:.3f}"),
     ("missed_attack_rate_last10", "Attacks missed", "{:.2%}"), ("false_alarm_rate_last10", "False alarms", "{:.2%}"),
     ("poisoned_mean_weight_share", "Poison share", "{:.1%}"), ("rare_mean_weight_share", "Rare share", "{:.1%}")]
v = lambda e, k: (e[k]["mean"] if isinstance(e.get(k), dict) else e.get(k)) if e else None
print("| Method | " + " | ".join(h for _, h, _ in K) + " |")
print("|" + "---|" * (len(K) + 1))
for m in ["multi_krum", "fedwgca", "fedavg", "fed_grab", "jiang_fixed", "jiang_dqn", "jiang_dqn_pre", "jiang_dqn_prexv",
          "fedavg_clean"]:
    if m not in N:
        continue
    cells = []
    for k, h, f in K:
        b = v(N[m], k); a = v(O.get(m), k)
        s = f.format(b) if b is not None else "–"
        if old and a is not None:
            s += f" ({f.format(a)})"
        cells.append(s)
    print(f"| {LAB[m]} | " + " | ".join(cells) + " |")
if old:
    print(f"\n(values in brackets: {old})")
