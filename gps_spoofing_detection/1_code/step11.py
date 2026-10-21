import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

W = 50
conds = ["straight", "curved", "random"]

data = pd.read_pickle("combined.pkl")
data = data.sort_values(["flight_id", "time_usec"]).reset_index(drop=True)

def add_features(g):
    g = g.copy()
    g["lat_delta"] = g["latitude_deg"].diff().fillna(0)
    g["lon_delta"] = g["longitude_deg"].diff().fillna(0)
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

print("Building features...")
data = pd.concat([add_features(g) for _, g in data.groupby("flight_id", sort=False)],
                 ignore_index=True)
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
y = data["attack_enabled"]
fid = data["flight_id"]
cond_idx = {c: np.where(data["condition"] == c)[0] for c in conds}

G = {
    "position": ["latitude_deg", "longitude_deg", "altitude", "pressure_altitude"],
    "orientation": ["orientation_x", "orientation_y", "orientation_z", "orientation_w"],
    "magnetic": ["magnetic_field_x", "magnetic_field_y", "magnetic_field_z"],
    "imu": ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
            "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"],
    "gps_velocity": ["velocity", "velocity_east", "velocity_north", "velocity_up"],
    "delta": ["lat_delta", "lon_delta"],
    "physics": ["speed_pos", "res_east", "res_north", "res_mag", "res_speed", "alt_diff"],
}
raw23 = (G["position"] + G["orientation"] + G["magnetic"] + G["imu"]
         + G["gps_velocity"] + G["delta"])
SETS = {
    "A_plus_physics": raw23 + G["physics"],
    "motion_plus_physics": G["imu"] + G["gps_velocity"] + G["delta"] + G["physics"],
    "B_refined14": ["velocity", "velocity_up", "speed_pos", "res_east", "res_north",
                    "res_mag", "res_speed", "alt_diff"] + G["imu"],
}
# Chosen by the nested selection in step 10 (training conditions only)
NESTED = {"straight": "A_plus_physics", "curved": "motion_plus_physics",
          "random": "A_plus_physics"}

MODELS = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=0, n_jobs=-1),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=100, max_depth=6, random_state=0,
        eval_metric="logloss", n_jobs=-1),
}

def flight_table(m, feats, idx):
    prob = m.predict_proba(data.iloc[idx][feats])[:, 1]
    df = pd.DataFrame({"fid": fid.iloc[idx].values, "prob": prob,
                       "y": y.iloc[idx].values})
    return df.groupby("fid").agg(score=("prob", "mean"), y=("y", "max")).reset_index(drop=True)

def balanced(ft, thr):
    pred = ft["score"] >= thr
    tpr = pred[ft["y"] == 1].mean()
    tnr = (~pred[ft["y"] == 0]).mean()
    return (tpr + tnr) / 2

def best_threshold(ft):
    grid = np.linspace(0.02, 0.98, 97)
    scores = [balanced(ft, t) for t in grid]
    return float(grid[int(np.argmax(scores))])

rows = []
t0 = time.time()
for model_name, make in MODELS.items():
    for plan in ["nested_choice", "B_refined14_exploratory"]:
        for held in conds:
            others = [c for c in conds if c != held]
            set_name = NESTED[held] if plan == "nested_choice" else "B_refined14"
            feats = SETS[set_name]

            # threshold from training conditions only (model trained on one, scored on the other)
            pooled = []
            for a, b in [(others[0], others[1]), (others[1], others[0])]:
                m = make()
                m.fit(data.iloc[cond_idx[a][::3]][feats], y.iloc[cond_idx[a][::3]])
                pooled.append(flight_table(m, feats, cond_idx[b]))
            thr = best_threshold(pd.concat(pooled, ignore_index=True))

            # final model trained on both training conditions, tested on unseen condition
            tr = np.concatenate([cond_idx[c] for c in others])[::3]
            m = make()
            m.fit(data.iloc[tr][feats], y.iloc[tr])
            ft = flight_table(m, feats, cond_idx[held])
            pred = ft["score"] >= thr
            tpr = float(pred[ft["y"] == 1].mean())
            fpr = float(pred[ft["y"] == 0].mean())
            rows.append({"model": model_name, "plan": plan, "held_out": held,
                         "features": set_name, "threshold": round(thr, 2),
                         "flight_auc": round(roc_auc_score(ft["y"], ft["score"]), 3),
                         "attacked_caught_TPR": round(tpr, 3),
                         "false_alarm_FPR": round(fpr, 3),
                         "balanced_acc": round((tpr + 1 - fpr) / 2, 3)})
            print(model_name, plan, held, "done", f"({time.time() - t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv("results_alarm_test.csv", index=False)
pd.set_option("display.width", 220)
print("\n===== PER HELD-OUT CONDITION =====")
print(res.to_string(index=False))
print("\n===== AVERAGE OVER THE 3 HELD-OUT CONDITIONS =====")
print(res.groupby(["model", "plan"])[["flight_auc", "attacked_caught_TPR",
                                      "false_alarm_FPR", "balanced_acc"]].mean().round(3).to_string())
print("\nSaved: results_alarm_test.csv")
print("ALL DONE")