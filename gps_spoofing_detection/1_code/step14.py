import glob
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from gps_common import add_features, FEATURES, analyze_flight

data = pd.read_pickle("combined.pkl")
data = data.sort_values(["flight_id", "time_usec"]).reset_index(drop=True)
data = pd.concat([add_features(g) for _, g in data.groupby("flight_id", sort=False)],
                 ignore_index=True)
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def make_rf():
    return RandomForestClassifier(n_estimators=100, max_depth=20,
                                  random_state=0, n_jobs=-1)

# ---- Honest flight-level scores: 5-fold cross-validation grouped by flight ----
oof = {}
idx = np.arange(len(data))
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(idx, groups=data["flight_id"])):
    tr = tr[::3]
    m = make_rf()
    m.fit(data.iloc[tr][FEATURES], data.iloc[tr]["attack_enabled"])
    prob = m.predict_proba(data.iloc[te][FEATURES])[:, 1]
    tmp = pd.DataFrame({"fid": data["flight_id"].iloc[te].values, "prob": prob})
    for fid, s in tmp.groupby("fid")["prob"].mean().items():
        oof[fid] = float(s)
    print("fold", k + 1, "done")

flights = data.groupby("flight_id").agg(
    condition=("condition", "first"), flight_type=("flight_type", "first"),
    res_mag=("res_mag", "mean")).reset_index()
flights["y"] = (flights["flight_type"] == "attacked").astype(int)
flights["score"] = flights["flight_id"].map(oof)
flights.to_csv("flight_scores_oof.csv", index=False)

# ---- Thresholds from NORMAL flights only (95th percentile) ----
thresholds = {"global": float(np.quantile(flights.loc[flights.y == 0, "score"], 0.95))}
for c in ["straight", "curved", "random"]:
    sub = flights[(flights.condition == c) & (flights.y == 0)]
    thresholds[c] = float(np.quantile(sub["score"], 0.95))
print("\nThresholds:", {k: round(v, 3) for k, v in thresholds.items()})

def rates(df, thr):
    tpr = (df.loc[df.y == 1, "score"] >= thr).mean()
    fpr = (df.loc[df.y == 0, "score"] >= thr).mean()
    return round(float(tpr), 3), round(float(fpr), 3)

print("\n===== FLIGHT-LEVEL RESULTS (cross-validated by flight) =====")
print("Pooled flight AUC:", round(roc_auc_score(flights.y, flights.score), 3))
tpr, fpr = rates(flights, thresholds["global"])
print("Global threshold, all flights: caught", tpr, "| false alarms", fpr)
for c in ["straight", "curved", "random"]:
    sub = flights[flights.condition == c]
    t1, f1 = rates(sub, thresholds["global"])
    t2, f2 = rates(sub, thresholds[c])
    print(f"{c:9s} AUC {roc_auc_score(sub.y, sub.score):.3f} | "
          f"global thr: caught {t1}, false alarms {f1} | "
          f"own-condition thr: caught {t2}, false alarms {f2}")

# ---- Final model on all flights + saved bundle ----
tr = np.arange(len(data))[::3]
final = make_rf()
final.fit(data.iloc[tr][FEATURES], data.iloc[tr]["attack_enabled"])
note = ("Prototype trained on simulated PX4-SITL flights. Features were refined "
        "after observing unseen-condition results, so reliability on new flight "
        "conditions is limited and must be checked with normal flights of the "
        "new condition before trusting alarms.")
joblib.dump({"model": final, "features": FEATURES, "thresholds": thresholds,
             "reliability_note": note}, "gps_detector.pkl")

# ---- Results file in a common structure for the integration layer ----
records = []
for _, r in flights.iterrows():
    records.append({
        "component": "GPS Spoofing Detection",
        "flight_id": r["flight_id"].replace("\\", "/"),
        "flight_condition": r["condition"],
        "detected_event": "GPS spoofing" if r["score"] >= thresholds["global"] else "Normal flight",
        "detected_event_condition_calibrated":
            "GPS spoofing" if r["score"] >= thresholds[r["condition"]] else "Normal flight",
        "confidence_score": round(float(r["score"]), 4),
        "threshold_global": round(thresholds["global"], 4),
        "evidence": {"mean_position_velocity_residual": round(float(r["res_mag"]), 4)},
        "actual_label": "attacked" if r["y"] == 1 else "normal",
    })
out = {"component": "GPS Spoofing Detection", "model": "RandomForest",
       "features": FEATURES, "reliability_note": note, "flight_results": records}
with open("gps_spoofing_results.json", "w") as f:
    json.dump(out, f, indent=2)

# ---- Demo: analyse one raw file with the saved detector ----
sample = glob.glob("Merged/**/*.csv", recursive=True)[0]
print("\nDemo on one file:")
print(json.dumps(analyze_flight(sample), indent=2))
print("\nSaved: gps_detector.pkl, gps_spoofing_results.json, flight_scores_oof.csv")
print("ALL DONE")