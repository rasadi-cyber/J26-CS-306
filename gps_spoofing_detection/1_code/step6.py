import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r1 = pd.read_csv("results.csv")
r1["experiment"] = "Exp 1 (all features)"
r2 = pd.read_csv("results_v2.csv")
r2["experiment"] = "Exp 2 (refined features)"
allr = pd.concat([r1, r2], ignore_index=True)
allr.to_csv("all_results.csv", index=False)

setups = ["Baseline (familiar)", "Unseen: straight", "Unseen: curved", "Unseen: random"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, exp in zip(axes, ["Exp 1 (all features)", "Exp 2 (refined features)"]):
    sub = allr[allr["experiment"] == exp].pivot(
        index="setup", columns="model", values="roc_auc").loc[setups]
    sub.plot(kind="bar", ax=ax, rot=20)
    ax.axhline(0.5, color="red", linestyle="--", label="Chance (0.5)")
    ax.set_title(exp)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1)
    ax.legend()
plt.tight_layout()
plt.savefig("roc_auc_comparison.png", dpi=200)
print("Saved: all_results.csv and roc_auc_comparison.png")