
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../results/sindy_grid_results.csv")
summary = df.groupby(["system", "noise", "obs_frac"]).agg(
    recall_mean=("recall", "mean"), recall_std=("recall", "std"),
    precision_mean=("precision", "mean"), precision_std=("precision", "std"),
).reset_index()

systems = summary["system"].unique()
fig, axes = plt.subplots(2, len(systems), figsize=(5 * len(systems), 8), sharey=True, sharex=True)

for col, name in enumerate(systems):
    sub = summary[summary["system"] == name]
    for obs_frac, group in sub.groupby("obs_frac"):
        group = group.sort_values("noise")
        axes[0, col].errorbar(group["noise"], group["recall_mean"], yerr=group["recall_std"],
                               marker="o", capsize=3, label=f"{int(obs_frac*100)}% obs")
        axes[1, col].errorbar(group["noise"], group["precision_mean"], yerr=group["precision_std"],
                               marker="o", capsize=3, label=f"{int(obs_frac*100)}% obs")
    axes[0, col].set_title(name)
    axes[1, col].set_xlabel("noise fraction")
    for row in (0, 1):
        axes[row, col].axhline(0.5, color="gray", ls="--", lw=0.5)

axes[0, 0].set_ylabel("recall (found the right terms?)")
axes[1, 0].set_ylabel("precision (avoided wrong terms?)")
axes[0, -1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("../figs/failure_boundary_v2.png", dpi=140)
print("Saved figs/failure_boundary_v2.png")
