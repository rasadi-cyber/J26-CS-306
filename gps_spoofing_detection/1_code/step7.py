import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

data = pd.read_pickle("combined.pkl")
data = data.sort_values(["flight_id", "time_usec"]).reset_index(drop=True)

W = 50

def add_features(g):
    g = g.copy()
    t = g["time_usec"] / 1e6
    lat0 = g["latitude_deg"].iloc[0]
    lon0 = g["longitude_deg"].iloc[0]
    x = (g["longitude_deg"] - lon0) * 111320 * np.cos(np.radians(lat0))
    y = (g["latitude_deg"] - lat0) * 110540
    dt = t - t.shift(W)
    g["vx_pos"] = (x - x.shift(W)) / dt
    g["vy_pos"] = (y - y.shift(W)) / dt
    g["speed_pos"] = np.sqrt(g["vx_pos"] ** 2 + g["vy_pos"] ** 2)
    ve = g["velocity_east"].rolling(W).mean()
    vn = g["velocity_north"].rolling(W).mean()
    g["res_east"] = g["vx_pos"] - ve
    g["res_north"] = g["vy_pos"] - vn
    g["res_mag"] = np.sqrt(g["res_east"] ** 2 + g["res_north"] ** 2)
    g["res_speed"] = g["speed_pos"] - g["velocity"].rolling(W).mean()
    g["alt_diff"] = g["altitude"] - g["pressure_altitude"]
    return g

data = pd.concat([add_features(g) for _, g in data.groupby("flight_id", sort=False)],
                 ignore_index=True)
data = data.replace([np.inf, -np.inf], np.nan).dropna()

features = ["velocity", "velocity_up", "speed_pos",
            "res_east", "res_north", "res_mag", "res_speed", "alt_diff",
            "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
            "linear_acceleration_x", "linear_acceleration_y",
            "linear_acceleration_z"]

# Same baseline split as step 5
flights = data[["flight_id", "condition", "flight_type"]].drop_duplicates()
strat = flights["condition"] + "_" + flights["flight_type"]
train_f, test_f = train_test_split(
    flights["flight_id"], test_size=0.3, random_state=42, stratify=strat)
tr = data["flight_id"].isin(train_f)
te = data["flight_id"].isin(test_f)
test = data[te].copy()

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=100, max_depth=6, random_state=42,
        eval_metric="logloss", n_jobs=-1),
}

records = []
timing = {}
for name, m in models.items():
    m.fit(data.loc[tr, features], data.loc[tr, "attack_enabled"])
    joblib.dump(m, f"{name}_model.pkl")

    t0 = time.time()
    prob = m.predict_proba(test[features])[:, 1]
    elapsed = time.time() - t0
    timing[name] = {
        "rows_predicted": int(len(test)),
        "prediction_seconds": round(elapsed, 3),
        "microseconds_per_row": round(elapsed / len(test) * 1e6, 2),
    }
    print(f"{name}: predicted {len(test)} rows in {elapsed:.2f} sec")

    test["prob"] = prob
    test["pred"] = (prob >= 0.5).astype(int)
    correct = 0
    total = 0
    for fid, g in test.groupby("flight_id"):
        attack_fraction = float(g["pred"].mean())
        detected = attack_fraction >= 0.5
        truth = g["flight_type"].iloc[0] == "attacked"
        correct += int(detected == truth)
        total += 1
        records.append({
            "component": "GPS Spoofing Detection",
            "model": name,
            "flight_id": fid.replace("\\", "/"),
            "flight_condition": g["condition"].iloc[0],
            "detected_event": "GPS spoofing" if detected else "Normal flight",
            "evidence": {
                "fraction_of_records_flagged": round(attack_fraction, 4),
                "mean_position_velocity_residual": round(float(g["res_mag"].mean()), 4),
            },
            "confidence": round(float(g["prob"].mean()), 4),
            "actual_label": "attacked" if truth else "normal",
        })
    print(f"{name}: flight-level accuracy on test flights = {correct}/{total}")

output = {
    "component": "GPS Spoofing Detection",
    "features_used": features,
    "prediction_timing": timing,
    "flight_results": records,
}
with open("gps_spoofing_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved: gps_spoofing_results.json, RandomForest_model.pkl, XGBoost_model.pkl")
print("ALL DONE")