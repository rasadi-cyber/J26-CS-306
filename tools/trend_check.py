"""Check every expected trend between methods (robust pole, rare-side pole, Jiang, DQN order) on a run's TEST
results; prints OK / ISSUE per rule. Works for any dataset: the rare class is read from the run.

  .venv/bin/python tools/trend_check.py RUN [--val]   (--val: last-10 validation instead of test, for pilots)
"""
import json,os,sys,numpy as np
run=sys.argv[1]; R=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"runs",run)
meta=json.load(open(f"{R}/meta.json")); S=meta["seeds"]
def load(m): return [json.load(open(f"{R}/{m}/seed_{s}/summary.json")) for s in S]
def say(m,role):
    v=[]
    for x in load(m):
        reg=x["general"]["mean_weight_share"]; v.append(x[role]["mean_weight_share"]/reg if reg else np.nan)
    return np.mean(v)
RARE=meta.get("rare_class","De-authentication")
HAVE=set(meta["methods"])
VAL="--val" in sys.argv
def last10_val(m):
    out=[]
    for s in S:
        rr=[json.loads(l) for l in open(f"{R}/{m}/seed_{s}/rounds.jsonl")]
        out.append([r["val"] for r in rr[-10:] if r.get("val")])
    return out
def rec(m,cls=None):
    cls=cls or RARE
    if VAL:
        return float(np.mean([np.mean([v["per_class"][cls]["recall"] for v in seed]) for seed in last10_val(m)]))
    cm=sum(np.array(x["confusion_last10"]) for x in load(m)); k=load(m)[0]["class_names"].index(cls); return cm[k,k]/cm[k].sum()
def met(m,k):
    if VAL: return float(np.mean([np.mean([v[k] for v in seed]) for seed in last10_val(m)]))
    return np.mean([x["test_last10"][k] for x in load(m)])
out=[]
def chk(name,ok,detail): out.append(("OK   " if ok else "ISSUE")+f" {name}: {detail}")
for m in [m for m in ("multi_krum","fedwgca") if m in HAVE]:
    chk(f"{m} (robust pole) cuts rare", say(m,"rare")<0.1, f"rare say {say(m,'rare'):.0%}")
    chk(f"{m} (robust pole) cuts/suppresses poison (say < 10% of a regular drone)", say(m,"poisoned")<0.1, f"poison say {say(m,'poisoned'):.0%}, missed {met(m,'missed_attack_rate'):.1%}")
for m in [m for m in ("fed_grab","fedavg") if m in HAVE]:
    chk(f"{m} (rare side) keeps rare ({RARE} recall >= 0.85)", rec(m)>=0.85, f"{RARE} recall {rec(m):.2f}")
    chk(f"{m} (rare side) lets poison in", say(m,"poisoned")>0.3, f"poison say {say(m,'poisoned'):.0%}, missed {met(m,'missed_attack_rate'):.1%}")
HAVE=set(meta["methods"])
J=[m for m in ["jiang_fixed","jiang_dqn","jiang_dqn_pre","jiang_dqn_prexv"] if m in HAVE]
for m in J:
    chk(f"{m} cuts poison", say(m,"poisoned")<0.01, f"poison say {say(m,'poisoned'):.0%}")
    chk(f"{m} keeps rare", rec(m)>=0.9, f"{RARE} recall {rec(m):.3f}, rare say {say(m,'rare'):.0%}")
best=min(met(m,"missed_attack_rate") for m in J)
others={m:met(m,"missed_attack_rate") for m in ("fedavg","multi_krum","fedwgca","fed_grab") if m in HAVE}
chk("Jiang misses fewest attacks", all(best<=v+1e-9 for v in others.values()), f"Jiang {best:.2%} vs "+", ".join(f"{k} {v:.2%}" for k,v in others.items()))
jr=max(rec(m) for m in J); orr={m:rec(m) for m in ("fedavg","multi_krum","fedwgca","fed_grab") if m in HAVE}
chk(f"Jiang highest {RARE} recall", all(jr>=v for v in orr.values()), f"Jiang {jr:.3f} vs "+", ".join(f"{k} {v:.2f}" for k,v in orr.items()))
jm=max(met(m,"macro_f1") for m in J); om={m:met(m,"macro_f1") for m in ("fedavg","multi_krum","fedwgca","fed_grab") if m in HAVE}
chk("Jiang highest macro F1", all(jm>v for v in om.values()), f"Jiang {jm:.3f} vs "+", ".join(f"{k} {v:.3f}" for k,v in om.items()))
jf=max(met(m,"rare_f1") for m in J); of={m:met(m,"rare_f1") for m in ("fedavg","fed_grab") if m in HAVE}
chk(f"Jiang {RARE} F1 >= rare-side methods", all(jf>=v for v in of.values()), f"Jiang {jf:.2f} vs "+", ".join(f"{k} {v:.2f}" for k,v in of.items()))
r=[rec(m) for m in J]; s=[say(m,'rare') for m in J]; f=[met(m,'macro_f1') for m in J]  # in J order
chk(f"DQN order: {RARE} recall "+" <= ".join(J), all(r[i]<=r[i+1]+1e-9 for i in range(len(J)-1)), " ".join(f"{x:.3f}" for x in r))
chk("DQN order: rare say", all(s[i]<=s[i+1]+1e-9 for i in range(len(J)-1)), " ".join(f"{x:.0%}" for x in s))
chk("DQN order: macro F1 (not required)", all(f[i]<=f[i+1]+0.003 for i in range(len(J)-1)), " ".join(f"{x:.3f}" for x in f))
print(("VALIDATION (last 10 rounds)" if VAL else "TEST (last 10 models)")+f" - rare class {RARE}");print("\n".join(out))
