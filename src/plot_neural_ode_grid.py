
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../results/neural_ode_grid_results.csv")
summary = df.groupby(["system", "noise", "obs_frac"]).agg(
    short_rmse_mean=("short_rmse", "mean"), short_rmse_std=("short_rmse", "std")
).reset_index()

systems = summary["system"].unique()
fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4))

for ax, name in zip(axes, systems):
    sub = summary[summary["system"] == name]
    for obs_frac, group in sub.groupby("obs_frac"):
        group = group.sort_values("noise")
        ax.errorbar(group["noise"], group["short_rmse_mean"], yerr=group["short_rmse_std"],
                     marker="o", capsize=3, label=f"{int(obs_frac*100)}% obs")
    ax.set_title(name)
    ax.set_xlabel("noise fraction")

axes[0].set_ylabel("short-horizon RMSE (predictability window)")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("../figs/neural_ode_grid.png", dpi=140)
print("Saved figs/neural_ode_grid.png")
