import os
import shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from gps_common import add_features, FEATURES, analyze_flight

conds = ["straight", "curved", "random"]
data = pd.read_pickle("combined.pkl")
data = data.sort_values(["flight_id", "time_usec"]).reset_index(drop=True)

flights = data[["flight_id", "condition", "flight_type"]].drop_duplicates()
rng = np.random.default_rng(42)
demo_ids = []
for c in conds:
    for ft in ["attacked", "normal"]:
        ids = flights[(flights.condition == c) &
                      (flights.flight_type == ft)]["flight_id"].values
        demo_ids += list(rng.choice(ids, size=2, replace=False))
print("Demo flights held out:", len(demo_ids))

train = data[~data["flight_id"].isin(demo_ids)].copy()
train = pd.concat([add_features(g) for _, g in train.groupby("flight_id", sort=False)],
                  ignore_index=True)
train = train.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def make_rf():
    return RandomForestClassifier(n_estimators=100, max_depth=20,
                                  random_state=0, n_jobs=-1)

# Flight scores from cross-validation grouped by flight
oof = {}
idx = np.arange(len(train))
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(idx, groups=train["flight_id"])):
    tr = tr[::3]
    m = make_rf()
    m.fit(train.iloc[tr][FEATURES], train.iloc[tr]["attack_enabled"])
    prob = m.predict_proba(train.iloc[te][FEATURES])[:, 1]
    tmp = pd.DataFrame({"fid": train["flight_id"].iloc[te].values, "prob": prob})
    for fid, s in tmp.groupby("fid")["prob"].mean().items():
        oof[fid] = float(s)
    print("fold", k + 1, "done")

fl = train.groupby("flight_id").agg(condition=("condition", "first"),
                                    flight_type=("flight_type", "first")).reset_index()
fl["score"] = fl["flight_id"].map(oof)
normal = fl[fl.flight_type == "normal"]
thresholds = {"global": float(np.quantile(normal["score"], 0.95))}
for c in conds:
    thresholds[c] = float(np.quantile(normal[normal.condition == c]["score"], 0.95))
print("Thresholds:", {k: round(v, 3) for k, v in thresholds.items()})

final = make_rf()
sub = np.arange(len(train))[::3]
final.fit(train.iloc[sub][FEATURES], train.iloc[sub]["attack_enabled"])
note = ("Demo model trained WITHOUT the 12 demo flights. Prototype trained on "
        "simulated PX4-SITL flights; thresholds come from normal flights of the "
        "selected condition. Reliability on new conditions is limited.")
joblib.dump({"model": final, "features": FEATURES, "thresholds": thresholds,
             "reliability_note": note}, "gps_detector_demo.pkl")

os.makedirs("demo_flights", exist_ok=True)
rows = []
for i, fid in enumerate(demo_ids, 1):
    info = flights[flights.flight_id == fid].iloc[0]
    dest = f"demo_flights/demo_{info['condition']}_{i:02d}.csv"
    shutil.copy(fid, dest)
    res = analyze_flight(dest, condition=info["condition"],
                         bundle_path="gps_detector_demo.pkl")
    actual = "GPS spoofing" if info["flight_type"] == "attacked" else "Normal flight"
    rows.append({"file": os.path.basename(dest), "condition": info["condition"],
                 "actual": actual, "detected": res["detected_event"],
                 "score": res["confidence_score"], "threshold": res["threshold_used"],
                 "correct": actual == res["detected_event"]})
out = pd.DataFrame(rows)
out.to_csv("demo_answers.csv", index=False)
pd.set_option("display.width", 200)
print("\n===== DEMO FLIGHTS (presenter only: do not show 'actual' first) =====")
print(out.to_string(index=False))
print("\nALL DONE")