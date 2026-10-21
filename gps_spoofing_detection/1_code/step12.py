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
NESTED = {"straight": "A_plus_physics", "curved": "motion_plus_physics",
          "random": "A_plus_physics"}

def make_model(name, weighted, y_tr):
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=0, n_jobs=-1,
            class_weight="balanced" if weighted else None)
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)) if weighted else 1.0
    return XGBClassifier(n_estimators=100, max_depth=6, random_state=0,
                         eval_metric="logloss", n_jobs=-1, scale_pos_weight=spw)

def flight_scores(m, feats, idx):
    prob = m.predict_proba(data.iloc[idx][feats])[:, 1]
    df = pd.DataFrame({"fid": fid.iloc[idx].values, "prob": prob,
                       "y": y.iloc[idx].values})
    return df.groupby("fid").agg(score=("prob", "mean"), y=("y", "max"))

rng = np.random.default_rng(0)
rows = []
t0 = time.time()
for model_name in ["RandomForest", "XGBoost"]:
    for plan in ["nested_choice", "B_refined14_exploratory"]:
        for held in conds:
            others = [c for c in conds if c != held]
            set_name = NESTED[held] if plan == "nested_choice" else "B_refined14"
            feats = SETS[set_name]
            tr = np.concatenate([cond_idx[c] for c in others])[::3]
            y_tr = y.iloc[tr]
            for weighted in [False, True]:
                m = make_model(model_name, weighted, y_tr)
                m.fit(data.iloc[tr][feats], y_tr)
                ft = flight_scores(m, feats, cond_idx[held])
                auc = roc_auc_score(ft["y"], ft["score"])
                att = ft.loc[ft["y"] == 1, "score"].values
                nor = ft.loc[ft["y"] == 0, "score"].values
                tprs, fprs = [], []
                for rep in range(20):
                    perm = rng.permutation(len(nor))
                    half = len(nor) // 2
                    calib = nor[perm[:half]]
                    test_n = nor[perm[half:]]
                    thr = np.quantile(calib, 0.95)
                    tprs.append((att > thr).mean())
                    fprs.append((test_n > thr).mean())
                rows.append({"model": model_name, "plan": plan, "held_out": held,
                             "class_weighting": weighted,
                             "flight_auc": round(auc, 3),
                             "calibrated_TPR": round(float(np.mean(tprs)), 3),
                             "calibrated_FPR": round(float(np.mean(fprs)), 3)})
            print(model_name, plan, held, "done", f"({time.time() - t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv("results_calibration.csv", index=False)
pd.set_option("display.width", 220)
print("\n===== PER HELD-OUT CONDITION =====")
print(res.to_string(index=False))
print("\n===== AVERAGE OVER THE 3 HELD-OUT CONDITIONS =====")
print(res.groupby(["model", "plan", "class_weighting"])[
    ["flight_auc", "calibrated_TPR", "calibrated_FPR"]].mean().round(3).to_string())
print("\nSaved: results_calibration.csv")
print("ALL DONE")