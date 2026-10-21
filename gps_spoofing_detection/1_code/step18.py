import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from gps_common import add_features, FEATURES

conds = ["straight", "curved", "random"]
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]

data = pd.read_pickle("combined.pkl")
data = data.sort_values(["flight_id", "time_usec"]).reset_index(drop=True)
print("Building features...")
data = pd.concat([add_features(g) for _, g in data.groupby("flight_id", sort=False)],
                 ignore_index=True)
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

RAW21 = ["pressure_altitude", "latitude_deg", "longitude_deg", "altitude",
         "velocity", "velocity_east", "velocity_north", "velocity_up",
         "orientation_x", "orientation_y", "orientation_z", "orientation_w",
         "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
         "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
         "magnetic_field_x", "magnetic_field_y", "magnetic_field_z"]
SETS = {"raw21": RAW21, "refined14 (exploratory)": FEATURES}

y = data["attack_enabled"].values
fid = data["flight_id"].values
fl = data[["flight_id", "condition", "flight_type"]].drop_duplicates().reset_index(drop=True)
fl["strat"] = fl["condition"] + "_" + fl["flight_type"]

def fit_eval(feats, train_flights, test_flights):
    tr = np.where(np.isin(fid, train_flights))[0][::3]
    te = np.where(np.isin(fid, test_flights))[0]
    m = XGBClassifier(n_estimators=100, max_depth=6, random_state=0,
                      eval_metric="logloss", n_jobs=-1)
    m.fit(data.iloc[tr][feats], y[tr])
    prob = m.predict_proba(data.iloc[te][feats])[:, 1]
    d = pd.DataFrame({"fid": fid[te], "p": prob, "y": y[te]})
    d = d.groupby("fid").agg(p=("p", "mean"), y=("y", "max"))
    return roc_auc_score(y[te], prob), roc_auc_score(d["y"], d["p"])

rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # Protocol A: flight-level split (test flights fixed per seed)
    tr_f, te_f = train_test_split(fl["flight_id"], test_size=0.3,
                                  random_state=seed, stratify=fl["strat"])
    tr_f = np.array(tr_f)
    order = rng.permutation(len(tr_f))
    for frac in FRACS:
        sub = tr_f[order[:max(int(len(tr_f) * frac), 6)]]
        for name, feats in SETS.items():
            r, f = fit_eval(feats, sub, np.array(te_f))
            rows.append({"protocol": "flight_level_split", "setting": "all",
                         "fraction": frac, "feature_set": name, "seed": seed,
                         "row_auc": r, "flight_auc": f})

    # Protocol B: leave-one-condition-out
    for held in conds:
        other = fl[fl.condition != held]["flight_id"].values
        test = fl[fl.condition == held]["flight_id"].values
        order = rng.permutation(len(other))
        for frac in FRACS:
            sub = other[order[:max(int(len(other) * frac), 6)]]
            for name, feats in SETS.items():
                r, f = fit_eval(feats, sub, test)
                rows.append({"protocol": "leave_one_condition_out", "setting": held,
                             "fraction": frac, "feature_set": name, "seed": seed,
                             "row_auc": r, "flight_auc": f})
    print("seed", seed, "done")

res = pd.DataFrame(rows)
res.to_csv("results_learning_curve.csv", index=False)
summ = res.groupby(["protocol", "feature_set", "fraction"])[["row_auc", "flight_auc"]].mean().round(3)
pd.set_option("display.width", 200)
print("\n===== LEARNING CURVE (mean over seeds; LOCO also averaged over held-out conditions) =====")
print(summ.to_string())

summ = summ.reset_index()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, proto in zip(axes, ["flight_level_split", "leave_one_condition_out"]):
    for name in SETS:
        s = summ[(summ["protocol"] == proto) & (summ["feature_set"] == name)]
        ax.plot(s["fraction"], s["flight_auc"], marker="o", label=name + " - flight AUC")
        ax.plot(s["fraction"], s["row_auc"], marker="x", linestyle="--",
                label=name + " - row AUC")
    ax.axhline(0.5, color="red", linestyle=":")
    ax.set_title(proto)
    ax.set_xlabel("Fraction of training flights used")
axes[0].set_ylabel("ROC-AUC")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=200)
print("\nSaved: results_learning_curve.csv, learning_curve.png")
print("ALL DONE")