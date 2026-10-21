import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

data = pd.read_pickle("combined.pkl")
data = data.sort_values(["flight_id", "time_usec"]).reset_index(drop=True)

W = 50   # window of 50 rows = about 2 seconds

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

parts = []
for fid, g in data.groupby("flight_id", sort=False):
    parts.append(add_features(g))
data = pd.concat(parts, ignore_index=True)
data = data.replace([np.inf, -np.inf], np.nan).dropna()
print("Rows after adding features:", data.shape[0])
print(pd.crosstab(data["condition"], data["attack_enabled"]))

# Features that do NOT depend on absolute location or heading
features = ["velocity", "velocity_up", "speed_pos",
            "res_east", "res_north", "res_mag", "res_speed", "alt_diff",
            "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
            "linear_acceleration_x", "linear_acceleration_y",
            "linear_acceleration_z"]
print("\nFeatures used:", features)

X = data[features]
y = data["attack_enabled"]

def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            eval_metric="logloss", n_jobs=-1),
    }

def evaluate(model, Xte, yte):
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    scores = {
        "accuracy": accuracy_score(yte, pred),
        "balanced_acc": balanced_accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, prob),
    }
    return scores, confusion_matrix(yte, pred, labels=[0, 1])

results = []
cms = {}

# ---- Baseline: split by flight file ----
flights = data[["flight_id", "condition", "flight_type"]].drop_duplicates()
strat = flights["condition"] + "_" + flights["flight_type"]
train_f, test_f = train_test_split(
    flights["flight_id"], test_size=0.3, random_state=42, stratify=strat)
tr = data["flight_id"].isin(train_f)
te = data["flight_id"].isin(test_f)
print("\nBaseline: train flights", len(train_f), "| test flights", len(test_f))

for name, m in get_models().items():
    t0 = time.time()
    m.fit(X[tr], y[tr])
    scores, cm = evaluate(m, X[te], y[te])
    results.append({"model": name, "setup": "Baseline (familiar)", **scores})
    cms[(name, "Baseline (familiar)")] = cm
    print(f"Done: {name} baseline ({time.time() - t0:.0f} sec)")
    if name == "RandomForest":
        imp = pd.Series(m.feature_importances_, index=features)
        print("\nRandom Forest feature importance:")
        print(imp.sort_values(ascending=False).round(3).to_string())

# ---- Leave-one-flight-condition-out ----
for held_out in ["straight", "curved", "random"]:
    tr = data["condition"] != held_out
    te = data["condition"] == held_out
    for name, m in get_models().items():
        t0 = time.time()
        m.fit(X[tr], y[tr])
        scores, cm = evaluate(m, X[te], y[te])
        setup = f"Unseen: {held_out}"
        results.append({"model": name, "setup": setup, **scores})
        cms[(name, setup)] = cm
        print(f"Done: {name} {setup} ({time.time() - t0:.0f} sec)")

res = pd.DataFrame(results)
res.to_csv("results_v2.csv", index=False)
print("\n===== RESULTS (Experiment 2) =====")
print(res.round(4).to_string(index=False))

metrics = ["accuracy", "balanced_acc", "precision", "recall", "f1", "roc_auc"]
base = res[res["setup"] == "Baseline (familiar)"].set_index("model")
deg = res[res["setup"].str.startswith("Unseen")].copy()
for mt in metrics:
    deg[mt + "_drop"] = deg.apply(lambda r: base.loc[r["model"], mt] - r[mt], axis=1)
deg = deg[["model", "setup"] + [m + "_drop" for m in metrics]]
deg.to_csv("degradation_v2.csv", index=False)
print("\n===== DEGRADATION (bigger = worse) =====")
print(deg.round(4).to_string(index=False))
print("\n===== AVERAGE DEGRADATION PER MODEL =====")
print(deg.groupby("model")[[m + "_drop" for m in metrics]].mean().round(4))

setups = ["Baseline (familiar)", "Unseen: straight", "Unseen: curved", "Unseen: random"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, name in enumerate(["RandomForest", "XGBoost"]):
    for j, s in enumerate(setups):
        sns.heatmap(cms[(name, s)], annot=True, fmt="d", cmap="Blues",
                    ax=axes[i][j], cbar=False,
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"])
        axes[i][j].set_title(f"{name}\n{s}")
        axes[i][j].set_xlabel("Predicted")
        axes[i][j].set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices_v2.png", dpi=200)
print("\nSaved: results_v2.csv, degradation_v2.csv, confusion_matrices_v2.png")
print("ALL DONE")