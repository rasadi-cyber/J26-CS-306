import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 1: feature-group ablation (XGBoost, leave-one-condition-out)
ab = pd.read_csv("results_ablation.csv")
m = ab.groupby("candidate")[["row_auc", "flight_auc"]].mean().sort_values("flight_auc")
fig, ax = plt.subplots(figsize=(8, 4.5))
m.plot(kind="barh", ax=ax)
ax.axvline(0.5, color="red", linestyle="--")
ax.set_xlabel("ROC-AUC (mean over the 3 unseen conditions)")
ax.set_title("Feature-group ablation")
plt.tight_layout()
plt.savefig("ablation_chart.png", dpi=200)

# Figure 2: calibrated alarm test (no class weighting)
cal = pd.read_csv("results_calibration.csv")
cal = cal[cal["class_weighting"] == False]
g = cal.groupby(["model", "plan"])[["calibrated_TPR", "calibrated_FPR"]].mean()
g.columns = ["Attacked flights caught", "False alarms on normal flights"]
fig, ax = plt.subplots(figsize=(9, 4.5))
g.plot(kind="bar", ax=ax, rot=15)
ax.set_ylim(0, 1)
ax.set_title("Normal-only calibration on unseen conditions (mean of 3)")
plt.tight_layout()
plt.savefig("calibration_chart.png", dpi=200)
print("Saved: ablation_chart.png, calibration_chart.png")