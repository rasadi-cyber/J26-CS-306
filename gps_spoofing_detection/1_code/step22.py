import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from gps_common import add_features, FEATURES

WINDOWS = [100, 200]
EXTRA = [f"{n}_{w}" for w in WINDOWS for n in ["speed_pos", "res_mag", "res_speed"]]
SEEDS = [0, 1, 2]
conds = ["straight", "curved", "random"]

def add_multi(g):
    g = add_features(g)
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
cond_arr = data["condition"].values
ftype = data.groupby("flight_id")["flight_type"].first()

VARIANTS = {"B_2s_window (14 features)": FEATURES,
            "C_multiscale (2, 4, 8 s)": FEATURES + EXTRA}
Xs = {n: data[f].to_numpy() for n, f in VARIANTS.items()}

rng = np.random.default_rng(0)
rows = []
for held in conds:
    te_idx = np.where(cond_arr == held)[0]
    tr_all = np.where(cond_arr != held)[0]
    for seed in SEEDS:
        tr = tr_all[seed::3]
        for vname in VARIANTS:
            X = Xs[vname]
            m = XGBClassifier(n_estimators=100, max_depth=6, random_state=seed,
                              eval_metric="logloss", n_jobs=-1)
            m.fit(X[tr], y_row[tr])
            prob = m.predict_proba(X[te_idx])[:, 1]
            row_auc = roc_auc_score(y_row[te_idx], prob)
            fs = pd.Series(prob).groupby(fid[te_idx]).mean()
            labels = (ftype.loc[fs.index] == "attacked").astype(int).values
            flight_auc = roc_auc_score(labels, fs.values)
            att = fs.values[labels == 1]
            nor = fs.values[labels == 0]
            tprs, fprs = [], []
            for rep in range(20):
                perm = rng.permutation(len(nor))
                half = len(nor) // 2
                thr = np.quantile(nor[perm[:half]], 0.95)   # normal flights only
                tprs.append((att > thr).mean())
                fprs.append((nor[perm[half:]] > thr).mean())
            tpr, fpr = float(np.mean(tprs)), float(np.mean(fprs))
            rows.append({"variant": vname, "held_out": held, "seed": seed,
                         "row_auc": row_auc, "flight_auc": flight_auc,
                         "attacks_caught": tpr, "false_alarms": fpr,
                         "balanced_acc": (tpr + 1 - fpr) / 2})
    print("held-out", held, "done")

res = pd.DataFrame(rows)
res.to_csv("results_unseen_multiscale.csv", index=False)
cols = ["row_auc", "flight_auc", "attacks_caught", "false_alarms", "balanced_acc"]
pd.set_option("display.width", 200)
print("\n===== UNSEEN FLIGHT TYPE, PER HELD-OUT TYPE (mean over 3 seeds) =====")
print(res.groupby(["variant", "held_out"])[cols].mean().round(3).to_string())
print("\n===== AVERAGE OVER THE 3 UNSEEN FLIGHT TYPES =====")
print(res.groupby("variant")[cols].mean().round(3).to_string())
print("\nSaved: results_unseen_multiscale.csv")
print("ALL DONE")