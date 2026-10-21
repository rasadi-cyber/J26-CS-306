import os
import shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from gps_common import add_features, FEATURES, analyze_flight

N_CAL = 20    # normal Random flights used ONLY to set the alarm line
N_DEMO = 4    # Random flights shown in the demo: 4 attacked + 4 normal

data = pd.read_pickle("combined.pkl")
data = data.sort_values(["flight_id", "time_usec"]).reset_index(drop=True)

def build(df):
    out = pd.concat([add_features(g) for _, g in df.groupby("flight_id", sort=False)],
                    ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

print("Building features...")
train = build(data[data["condition"].isin(["straight", "curved"])])
rand = build(data[data["condition"] == "random"])

# 1) Train on Straight + Curved ONLY. Random flights are never used for training.
model = RandomForestClassifier(n_estimators=100, max_depth=20,
                               random_state=0, n_jobs=-1)
sub = np.arange(len(train))[::3]
model.fit(train.iloc[sub][FEATURES], train.iloc[sub]["attack_enabled"])
print("Model trained on Straight + Curved only")

# 2) Score every Random flight
rand["prob"] = model.predict_proba(rand[FEATURES])[:, 1]
fl = rand.groupby("flight_id").agg(score=("prob", "mean"),
                                   flight_type=("flight_type", "first")).reset_index()
fl["y"] = (fl["flight_type"] == "attacked").astype(int)

# 3) Pick flights with a fixed seed (no cherry-picking)
rng = np.random.default_rng(42)
normal_ids = [str(x) for x in rng.permutation(fl[fl.y == 0]["flight_id"].values)]
attack_ids = [str(x) for x in rng.permutation(fl[fl.y == 1]["flight_id"].values)]
cal_ids = normal_ids[:N_CAL]
demo_normal = normal_ids[N_CAL:N_CAL + N_DEMO]
demo_attack = attack_ids[:N_DEMO]

# 4) Alarm line from normal Random flights only (normal-only calibration)
cal_scores = fl[fl["flight_id"].isin(cal_ids)]["score"]
thr = float(np.quantile(cal_scores, 0.95))
print(f"Alarm line from {N_CAL} normal Random flights: {thr:.3f}")

# 5) A bigger honest check: all Random flights NOT used for calibration
rest = fl[~fl["flight_id"].isin(cal_ids)]
pred = rest["score"] >= thr
n_att = int((rest.y == 1).sum())
n_nor = int((rest.y == 0).sum())
caught = int(((rest.y == 1) & pred).sum())
false_alarms = int(((rest.y == 0) & pred).sum())
print("\n===== UNSEEN CONDITION (Random) =====")
print("Flight AUC over all 144 Random flights:",
      round(roc_auc_score(fl["y"], fl["score"]), 3))
print(f"Attacked flights caught: {caught} of {n_att}")
print(f"False alarms on normal flights: {false_alarms} of {n_nor}")

# 6) Save the unseen-condition demo model
note = ("UNSEEN-CONDITION DEMO: trained only on Straight and Curved flights; "
        "Random flights were never used for training. The alarm line comes from "
        f"{N_CAL} normal Random flights. Simulated data; features are exploratory; "
        "reliability is limited.")
joblib.dump({"model": model, "features": FEATURES,
             "thresholds": {"global": thr, "random": thr},
             "reliability_note": note}, "gps_detector_unseen.pkl")

# 7) Demo flights
os.makedirs("demo_flights_unseen", exist_ok=True)
rows = []
for i, fid in enumerate(demo_attack + demo_normal, 1):
    dest = f"demo_flights_unseen/unseen_random_{i:02d}.csv"
    shutil.copy(fid, dest)
    res = analyze_flight(dest, condition="random",
                         bundle_path="gps_detector_unseen.pkl")
    actual = "GPS spoofing" if fid in demo_attack else "Normal flight"
    rows.append({"file": os.path.basename(dest), "actual": actual,
                 "detected": res["detected_event"],
                 "score": res["confidence_score"],
                 "threshold": res["threshold_used"],
                 "correct": actual == res["detected_event"]})
out = pd.DataFrame(rows)
out.to_csv("demo_answers_unseen.csv", index=False)
pd.set_option("display.width", 200)
print("\n===== DEMO FLIGHTS (presenter only) =====")
print(out.to_string(index=False))
print("\nALL DONE")