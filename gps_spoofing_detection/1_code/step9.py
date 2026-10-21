import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

SEEDS = [0, 1, 2]
TRAIN_STEP = 3   # train on every 3rd row to save time (test uses all rows)
W = 50

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
print("Rows used:", len(data))

FEATURE_SETS = {
    "A_raw23": ['pressure_altitude', 'latitude_deg', 'longitude_deg', 'altitude',
                'velocity', 'velocity_east', 'velocity_north', 'velocity_up',
                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
                'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z',
                'lat_delta', 'lon_delta'],
    "B_refined14": ['velocity', 'velocity_up', 'speed_pos', 'res_east', 'res_north',
                    'res_mag', 'res_speed', 'alt_diff',
                    'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                    'linear_acceleration_x', 'linear_acceleration_y',
                    'linear_acceleration_z'],
}
MODELS = {
    "RandomForest": lambda s: RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=s, n_jobs=-1),
    "XGBoost": lambda s: XGBClassifier(
        n_estimators=100, max_depth=6, random_state=s,
        eval_metric="logloss", n_jobs=-1),
}

def score(model, feats, te_idx):
    X = data.iloc[te_idx][feats]
    yt = y.iloc[te_idx]
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    df = pd.DataFrame({"fid": data["flight_id"].iloc[te_idx].values,
                       "prob": prob, "y": yt.values})
    fl = df.groupby("fid").agg(prob=("prob", "mean"), y=("y", "max"))
    flight_auc = roc_auc_score(fl["y"], fl["prob"]) if fl["y"].nunique() > 1 else np.nan
    return (roc_auc_score(yt, prob), balanced_accuracy_score(yt, pred),
            f1_score(yt, pred, zero_division=0), flight_auc)

records = []
def run(protocol, setting, tr_idx, te_idx, seed):
    tr_idx = tr_idx[::TRAIN_STEP]
    for fs_name, feats in FEATURE_SETS.items():
        for m_name, make in MODELS.items():
            m = make(seed)
            m.fit(data.iloc[tr_idx][feats], y.iloc[tr_idx])
            r_auc, bal, f1, f_auc = score(m, feats, te_idx)
            records.append({"protocol": protocol, "setting": setting,
                            "feature_set": fs_name, "model": m_name, "seed": seed,
                            "row_auc": r_auc, "bal_acc": bal, "f1": f1,
                            "flight_auc": f_auc})

flights = data[["flight_id", "condition", "flight_type"]].drop_duplicates()
strat = flights["condition"] + "_" + flights["flight_type"]
idx_all = np.arange(len(data))

t0 = time.time()
for seed in SEEDS:
    # Protocol 1: random row split (ignores which flight a row came from)
    tr, te = train_test_split(idx_all, test_size=0.3, random_state=seed, stratify=y)
    run("1_random_row_split", "all", tr, te, seed)
    print(f"seed {seed}: protocol 1 done ({time.time()-t0:.0f}s)")

    # Protocol 2: flight-level split
    train_f, test_f = train_test_split(flights["flight_id"], test_size=0.3,
                                       random_state=seed, stratify=strat)
    tr = np.where(data["flight_id"].isin(train_f))[0]
    te = np.where(data["flight_id"].isin(test_f))[0]
    run("2_flight_level_split", "all", tr, te, seed)
    print(f"seed {seed}: protocol 2 done ({time.time()-t0:.0f}s)")

    # Protocol 3: leave-one-flight-condition-out
    for held in ["straight", "curved", "random"]:
        tr = np.where(data["condition"] != held)[0]
        te = np.where(data["condition"] == held)[0]
        run("3_leave_one_condition_out", held, tr, te, seed)
    print(f"seed {seed}: protocol 3 done ({time.time()-t0:.0f}s)")

res = pd.DataFrame(records)
res.to_csv("results_protocols.csv", index=False)

pd.set_option("display.width", 220)
cols = ["row_auc", "bal_acc", "f1", "flight_auc"]
print("\n===== MEAN (std) OVER SEEDS AND SETTINGS =====")
print(res.groupby(["feature_set", "model", "protocol"])[cols].agg(["mean", "std"]).round(3).to_string())
print("\n===== LEAVE-ONE-CONDITION-OUT, PER HELD-OUT CONDITION =====")
loco = res[res["protocol"] == "3_leave_one_condition_out"]
print(loco.groupby(["feature_set", "model", "setting"])[cols].mean().round(3).to_string())

# Chart: row-level ROC-AUC per protocol
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, fs in zip(axes, FEATURE_SETS):
    sub = res[res["feature_set"] == fs].groupby(["protocol", "model"])["row_auc"].mean().unstack()
    sub.plot(kind="bar", ax=ax, rot=15)
    ax.axhline(0.5, color="red", linestyle="--")
    ax.set_title(fs)
    ax.set_ylabel("Row-level ROC-AUC")
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("protocol_comparison.png", dpi=200)
print("\nSaved: results_protocols.csv, protocol_comparison.png")
print("ALL DONE")