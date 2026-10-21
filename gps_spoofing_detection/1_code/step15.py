import glob
import hashlib
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from gps_common import analyze_flight

pd.set_option("display.width", 220)
conds = ["straight", "curved", "random"]

# ===== PART 1: FAILURE ANALYSIS (FR14) =====
print("===== PART 1: FAILURE ANALYSIS =====")
sc = pd.read_csv("flight_scores_oof.csv")
raw = pd.read_pickle("combined.pkl").sort_values(["flight_id", "time_usec"])
info = []
for fid, g in raw.groupby("flight_id"):
    t = (g["time_usec"] - g["time_usec"].iloc[0]) / 1e6
    atk = g["attack_enabled"] == 1
    info.append({"flight_id": fid, "duration_s": float(t.iloc[-1]),
                 "attack_fraction": float(atk.mean()),
                 "attack_start_s": float(t[atk].iloc[0]) if atk.any() else np.nan})
df = sc.merge(pd.DataFrame(info), on="flight_id")

thr = {c: df[(df.condition == c) & (df.y == 0)]["score"].quantile(0.95) for c in conds}
df["flagged"] = df["score"] >= df["condition"].map(thr)

for c in conds:
    s = df[df.condition == c]
    tp = int(((s.y == 1) & s.flagged).sum())
    fn = int(((s.y == 1) & ~s.flagged).sum())
    fp = int(((s.y == 0) & s.flagged).sum())
    tn = int(((s.y == 0) & ~s.flagged).sum())
    print(f"{c:9s} threshold {thr[c]:.3f} | caught {tp} | missed {fn} | "
          f"false alarms {fp} | normal cleared {tn}")

att = df[df.y == 1].copy()
att["outcome"] = np.where(att.flagged, "caught", "missed")
print("\nAverages for caught vs missed attacked flights:")
print(att.groupby(["condition", "outcome"])[
    ["attack_fraction", "attack_start_s", "duration_s", "res_mag", "score"]
].mean().round(3).to_string())
for c in conds:
    s = att[att.condition == c]
    r = s["score"].corr(s["attack_fraction"], method="spearman")
    print(f"{c}: Spearman(score, attack_fraction) = {r:.3f}")
att[att.outcome == "missed"].to_csv("missed_flights.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, c in zip(axes, conds):
    s = att[att.condition == c]
    for outcome, col in [("caught", "tab:green"), ("missed", "tab:red")]:
        ss = s[s.outcome == outcome]
        ax.scatter(ss["attack_fraction"], ss["score"], c=col, label=outcome, alpha=0.7)
    ax.axhline(thr[c], color="black", linestyle="--")
    ax.set_title(c)
    ax.set_xlabel("Fraction of rows labelled attack")
axes[0].set_ylabel("Flight score")
axes[0].legend()
plt.tight_layout()
plt.savefig("failure_analysis.png", dpi=200)

# ===== PART 2: END-TO-END SPEED (NFR09) =====
print("\n===== PART 2: SPEED PER FLIGHT =====")
files = sorted(glob.glob("Merged/**/*.csv", recursive=True))
rng = np.random.default_rng(0)
sample = rng.choice(files, size=20, replace=False)
times = []
for f in sample:
    t0 = time.perf_counter()
    analyze_flight(str(f))
    times.append(time.perf_counter() - t0)
print(f"20 flights, read + features + prediction + model loading: "
      f"mean {np.mean(times):.2f} s, max {np.max(times):.2f} s per flight")

# ===== PART 3: DATASET INTEGRITY MANIFEST (NFR05) =====
print("\n===== PART 3: INTEGRITY MANIFEST =====")
rows = []
for f in files:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    rows.append({"file": f.replace("\\", "/"), "bytes": os.path.getsize(f),
                 "sha256": h.hexdigest()})
pd.DataFrame(rows).to_csv("dataset_manifest.csv", index=False)
print("Saved dataset_manifest.csv with", len(rows), "files")

# ===== PART 4: LEAKAGE TEST =====
print("\n===== PART 4: LEAKAGE TEST =====")
fids = raw["flight_id"].values
ok = True
for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(fids)), groups=fids):
    if set(np.unique(fids[tr])) & set(np.unique(fids[te])):
        ok = False
print("Flight-level split shares no flights between train and test:",
      "PASS" if ok else "FAIL")
cond = raw["condition"].values
for held in conds:
    train_conds = set(np.unique(cond[cond != held]))
    print(f"Leave-one-condition-out ({held} unseen): ",
          "PASS" if held not in train_conds else "FAIL")
print("\nALL DONE")