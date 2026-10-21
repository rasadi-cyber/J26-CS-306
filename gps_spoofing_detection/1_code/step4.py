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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

data = pd.read_pickle("combined.pkl")

# ---- Derived features (latitude/longitude change inside each flight) ----
data = data.sort_values(["flight_id", "time_usec"])
data["lat_delta"] = data.groupby("flight_id")["latitude_deg"].diff().fillna(0)
data["lon_delta"] = data.groupby("flight_id")["longitude_deg"].diff().fillna(0)

# ---- Feature selection (time_usec is removed on purpose) ----
drop_cols = ["time_usec", "attack_enabled", "condition", "flight_type", "flight_id"]
features = [c for c in data.columns if c not in drop_cols]
print("Number of features:", len(features))
print(features)

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
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, prob),
    }
    return scores, confusion_matrix(yte, pred, labels=[0, 1])

results = []
cms = {}

# ---- 1. BASELINE: split by flight file, every condition appears in train and test ----
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

# ---- 2. LEAVE-ONE-FLIGHT-CONDITION-OUT ----
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

# ---- 3. RESULTS TABLE ----
res = pd.DataFrame(results)
res.to_csv("results.csv", index=False)
print("\n===== RESULTS =====")
print(res.round(4).to_string(index=False))

# ---- 4. PERFORMANCE DEGRADATION = familiar - unseen ----
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
base = res[res["setup"] == "Baseline (familiar)"].set_index("model")
deg = res[res["setup"].str.startswith("Unseen")].copy()
for mt in metrics:
    deg[mt + "_drop"] = deg.apply(lambda r: base.loc[r["model"], mt] - r[mt], axis=1)
deg = deg[["model", "setup"] + [m + "_drop" for m in metrics]]
deg.to_csv("degradation.csv", index=False)
print("\n===== DEGRADATION (bigger = worse) =====")
print(deg.round(4).to_string(index=False))

print("\n===== AVERAGE DEGRADATION PER MODEL (lower = more robust) =====")
print(deg.groupby("model")[[m + "_drop" for m in metrics]].mean().round(4))

# ---- 5. CONFUSION MATRICES ----
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
plt.savefig("confusion_matrices.png", dpi=200)
print("\nSaved: results.csv, degradation.csv, confusion_matrices.png")
print("ALL DONE")