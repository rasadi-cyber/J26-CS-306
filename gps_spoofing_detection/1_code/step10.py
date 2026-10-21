import time
import numpy as np
import pandas as pd
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
CANDIDATES = {
    "A_raw23": raw23,
    "A_plus_physics": raw23 + G["physics"],
    "no_position": G["orientation"] + G["magnetic"] + G["imu"] + G["gps_velocity"] + G["delta"],
    "motion_only": G["imu"] + G["gps_velocity"] + G["delta"],
    "motion_plus_physics": G["imu"] + G["gps_velocity"] + G["delta"] + G["physics"],
    "B_refined14": ["velocity", "velocity_up", "speed_pos", "res_east", "res_north",
                    "res_mag", "res_speed", "alt_diff"] + G["imu"],
}

def fit_score(feats, tr_idx, te_idx):
    m = XGBClassifier(n_estimators=100, max_depth=6, random_state=0,
                      eval_metric="logloss", n_jobs=-1)
    m.fit(data.iloc[tr_idx][feats], y.iloc[tr_idx])
    prob = m.predict_proba(data.iloc[te_idx][feats])[:, 1]
    yt = y.iloc[te_idx].values
    df = pd.DataFrame({"fid": fid.iloc[te_idx].values, "prob": prob, "y": yt})
    fl = df.groupby("fid").agg(prob=("prob", "mean"), y=("y", "max"))
    return roc_auc_score(yt, prob), roc_auc_score(fl["y"], fl["prob"])

# ---------- Part 1: ablation (leave-one-condition-out) ----------
t0 = time.time()
rows = []
for held in conds:
    others = [c for c in conds if c != held]
    tr_all = np.concatenate([cond_idx[c] for c in others])
    for off in [0, 1, 2]:
        for name, feats in CANDIDATES.items():
            r, f = fit_score(feats, tr_all[off::3], cond_idx[held])
            rows.append({"candidate": name, "held_out": held, "offset": off,
                         "row_auc": r, "flight_auc": f})
    print(f"ablation: held-out {held} done ({time.time() - t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv("results_ablation.csv", index=False)
pd.set_option("display.width", 220)
print("\n===== ABLATION: mean over 3 training subsamples =====")
print(res.groupby(["candidate", "held_out"])[["row_auc", "flight_auc"]]
      .mean().round(3).unstack().to_string())
print("\n===== ABLATION: average over the 3 held-out conditions =====")
print(res.groupby("candidate")[["row_auc", "flight_auc"]].mean().round(3).to_string())

# ---------- Part 2: nested selection (uses only training conditions) ----------
print("\n===== NESTED SELECTION =====")
nested = []
for held in conds:
    others = [c for c in conds if c != held]
    inner = {}
    for name, feats in CANDIDATES.items():
        sc = []
        for a, b in [(others[0], others[1]), (others[1], others[0])]:
            r, f = fit_score(feats, cond_idx[a][::3], cond_idx[b])
            sc.append((r + f) / 2)
        inner[name] = float(np.mean(sc))
    chosen = max(inner, key=inner.get)
    tr = np.concatenate([cond_idx[c] for c in others])[::3]
    r, f = fit_score(CANDIDATES[chosen], tr, cond_idx[held])
    nested.append({"held_out": held, "chosen": chosen, "outer_row_auc": r,
                   "outer_flight_auc": f})
    print(held, "| inner scores:", {k: round(v, 3) for k, v in inner.items()})
    print("   chosen:", chosen, "| held-out row AUC", round(r, 3),
          "| flight AUC", round(f, 3))

pd.DataFrame(nested).to_csv("results_nested.csv", index=False)
print("\nSaved: results_ablation.csv, results_nested.csv")
print("ALL DONE")