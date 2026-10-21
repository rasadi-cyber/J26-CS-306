import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from gps_common import add_features, FEATURES

WINDOWS = [100, 200]          # 4 s and 8 s at about 25 Hz
EXTRA = [f"{n}_{w}" for w in WINDOWS for n in ["speed_pos", "res_mag", "res_speed"]]
SEEDS = [0, 1, 2]
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)
conds = ["straight", "curved", "random"]

def add_multi(g):
    g = add_features(g)       # original 2 s window features
    t = g["time_usec"] / 1e6
    lat0 = g["latitude_deg"].iloc[0]
    lon0 = g["longitude_deg"].iloc[0]
    x = (g["longitude_deg"] - lon0) * 111320 * np.cos(np.radians(lat0))
    yy = (g["latitude_deg"] - lat0) * 110540
    for w in WINDOWS:
        dt = t - t.shift(w)
        vx = (x - x.shift(w)) / dt
        vy = (yy - yy.shift(w)) / dt
        sp = np.sqrt(vx ** 2 + vy ** 2)
        re = vx - g["velocity_east"].rolling(w).mean()
        rn = vy - g["velocity_north"].rolling(w).mean()
        g[f"speed_pos_{w}"] = sp
        g[f"res_mag_{w}"] = np.sqrt(re ** 2 + rn ** 2)
        g[f"res_speed_{w}"] = sp - g["velocity"].rolling(w).mean()
    return g

print("Building features...")
data = pd.read_pickle("combined.pkl")
data = data.sort_values(["flight_id", "time_usec"]).reset_index(drop=True)
data = pd.concat([add_multi(g) for _, g in data.groupby("flight_id", sort=False)],
                 ignore_index=True)
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

y_row = data["attack_enabled"].values
fid = data["flight_id"].values
rows_of = data.groupby("flight_id").indices

flights = (data.groupby("flight_id")
           .agg(condition=("condition", "first"), flight_type=("flight_type", "first"))
           .reset_index())
flights["y"] = (flights["flight_type"] == "attacked").astype(int)
flights["strat_code"] = (flights["condition"] + "_" + flights["flight_type"]) \
    .astype("category").cat.codes
print("Flights:", len(flights),
      "| accuracy of always guessing the majority class:",
      round(max(flights["y"].mean(), 1 - flights["y"].mean()), 3))

VARIANTS = {"B_2s_window (14 features)": FEATURES,
            "C_multiscale (2, 4, 8 s)": FEATURES + EXTRA}
Xs = {name: data[feats].to_numpy() for name, feats in VARIANTS.items()}

def rows_for(flist):
    return np.concatenate([rows_of[f] for f in flist])

def fit_model(X, train_flights, seed):
    tr = rows_for(train_flights)[::3]
    m = XGBClassifier(n_estimators=100, max_depth=6, random_state=seed,
                      eval_metric="logloss", n_jobs=-1)
    m.fit(X[tr], y_row[tr])
    return m

def score_flights(m, X, flist):
    idx = rows_for(flist)
    p = m.predict_proba(X[idx])[:, 1]
    return pd.Series(p).groupby(fid[idx]).mean()

def best_threshold(scores, labels):
    best, best_t = -1.0, 0.5
    for t in GRID:
        pred = scores >= t
        tpr = pred[labels == 1].mean() if (labels == 1).any() else 0.0
        tnr = (~pred[labels == 0]).mean() if (labels == 0).any() else 0.0
        b = (tpr + tnr) / 2
        if b > best:
            best, best_t = b, t
    return float(best_t)

records = []
for seed in SEEDS:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for vname in VARIANTS:
        X = Xs[vname]
        for tr_i, te_i in outer.split(flights, flights["strat_code"]):
            tr_f = flights.iloc[tr_i].reset_index(drop=True)
            te_f = flights.iloc[te_i].reset_index(drop=True)

            # Inner cross-validation on TRAINING flights only: gives scores
            # from which the alarm line of each flight type is chosen.
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            oof = {}
            for a, b in inner.split(tr_f, tr_f["strat_code"]):
                m = fit_model(X, tr_f["flight_id"].iloc[a].values, seed)
                s = score_flights(m, X, tr_f["flight_id"].iloc[b].values)
                oof.update(s.to_dict())
            tr_f["oof"] = tr_f["flight_id"].map(oof)
            thr = {}
            for c in conds:
                sub = tr_f[tr_f["condition"] == c]
                thr[c] = best_threshold(sub["oof"].values, sub["y"].values)

            # Final model for this fold, scored on flights it has never seen
            m = fit_model(X, tr_f["flight_id"].values, seed)
            s = score_flights(m, X, te_f["flight_id"].values)
            for _, r in te_f.iterrows():
                records.append({"variant": vname, "seed": seed,
                                "flight_id": r["flight_id"],
                                "condition": r["condition"], "y": int(r["y"]),
                                "score": float(s[r["flight_id"]]),
                                "threshold": thr[r["condition"]]})
    print("seed", seed, "done")

res = pd.DataFrame(records)
res["pred"] = (res["score"] >= res["threshold"]).astype(int)
res["correct"] = (res["pred"] == res["y"]).astype(int)
res.to_csv("results_flight_cv.csv", index=False)

def summarize(df):
    tpr = df.loc[df["y"] == 1, "pred"].mean()
    fpr = df.loc[df["y"] == 0, "pred"].mean()
    aucs = [roc_auc_score(g["y"], g["score"]) for _, g in df.groupby("condition")]
    return {"accuracy": df["correct"].mean(), "balanced_acc": (tpr + 1 - fpr) / 2,
            "attacks_caught": tpr, "false_alarms": fpr,
            "mean_auc_per_condition": float(np.mean(aucs))}

rows = []
for (v, sd), df in res.groupby(["variant", "seed"]):
    rows.append({"variant": v, "seed": sd, **summarize(df)})
tab = pd.DataFrame(rows)
cols = ["accuracy", "balanced_acc", "attacks_caught", "false_alarms",
        "mean_auc_per_condition"]
pd.set_option("display.width", 200)
print("\n===== WHOLE-FLIGHT RESULTS (5-fold by flight, mean and std over 3 seeds) =====")
print(tab.groupby("variant")[cols].agg(["mean", "std"]).round(3).to_string())
print("\n===== ACCURACY PER FLIGHT TYPE =====")
print(res.groupby(["variant", "condition"])["correct"].mean().round(3).unstack().to_string())
print("\nSaved: results_flight_cv.csv")
print("ALL DONE")