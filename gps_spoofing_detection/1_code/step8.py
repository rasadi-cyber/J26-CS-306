import pandas as pd

data = pd.read_pickle("combined.pkl")
rows = []
for fid, g in data.groupby("flight_id"):
    g = g.sort_values("time_usec")
    t = (g["time_usec"] - g["time_usec"].iloc[0]) / 1e6
    atk = g["attack_enabled"] == 1
    rows.append({
        "condition": g["condition"].iloc[0],
        "flight_type": g["flight_type"].iloc[0],
        "duration_s": t.iloc[-1],
        "attack_fraction": atk.mean(),
        "attack_start_s": t[atk].iloc[0] if atk.any() else float("nan"),
    })

f = pd.DataFrame(rows)
f.to_csv("flight_summary.csv", index=False)

summary = f.groupby(["condition", "flight_type"]).agg(
    flights=("duration_s", "size"),
    duration_mean=("duration_s", "mean"),
    atk_fraction_mean=("attack_fraction", "mean"),
    atk_fraction_min=("attack_fraction", "min"),
    atk_fraction_max=("attack_fraction", "max"),
    atk_start_mean=("attack_start_s", "mean"),
    atk_start_min=("attack_start_s", "min"),
    atk_start_max=("attack_start_s", "max"),
).round(3)
print(summary.to_string())

normal = f[f["flight_type"] == "normal"]
print("\nNormal flights containing any attack rows:", int((normal["attack_fraction"] > 0).sum()))