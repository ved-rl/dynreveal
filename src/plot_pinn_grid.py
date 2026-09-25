
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../results/pinn_grid_results.csv")
summary = df.groupby(["system", "noise", "obs_frac"]).agg(
    mean_rel_error_mean=("mean_rel_error", "mean"),
    mean_rel_error_std=("mean_rel_error", "std"),
).reset_index()

systems = summary["system"].unique()
fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4))

for ax, name in zip(axes, systems):
    sub = summary[summary["system"] == name]
    for obs_frac, group in sub.groupby("obs_frac"):
        group = group.sort_values("noise")
        ax.errorbar(group["noise"], group["mean_rel_error_mean"], yerr=group["mean_rel_error_std"],
                     marker="o", capsize=3, label=f"{int(obs_frac*100)}% obs")
    ax.set_title(name)
    ax.set_xlabel("noise fraction")
    ax.set_yscale("log")

axes[0].set_ylabel("mean relative parameter error (log scale)")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("../figs/pinn_grid.png", dpi=140)
print("Saved figs/pinn_grid.png")
