
import pandas as pd

print("=" * 70)
print("SINDy (5 seeds)")
print("=" * 70)
df = pd.read_csv("../results/sindy_grid_results.csv")
summary = df.groupby(["system", "noise", "obs_frac"]).agg(
    recall_mean=("recall", "mean"), recall_std=("recall", "std"),
    precision_mean=("precision", "mean"), precision_std=("precision", "std"),
    n_seeds=("seed", "count"),
).reset_index()
summary.to_csv("../results/sindy_summary_stats.csv", index=False)
print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("Neural ODE (5 seeds)")
print("=" * 70)
df = pd.read_csv("../results/neural_ode_grid_results.csv")
summary = df.groupby(["system", "noise", "obs_frac"]).agg(
    short_rmse_mean=("short_rmse", "mean"), short_rmse_std=("short_rmse", "std"),
    n_seeds=("seed", "count"),
).reset_index()
summary.to_csv("../results/neural_ode_summary_stats.csv", index=False)
print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("PINN (3 seeds)")
print("=" * 70)
df = pd.read_csv("../results/pinn_grid_results.csv")
summary = df.groupby(["system", "noise", "obs_frac"]).agg(
    mean_rel_error_mean=("mean_rel_error", "mean"),
    mean_rel_error_std=("mean_rel_error", "std"),
    n_seeds=("seed", "count"),
).reset_index()
summary.to_csv("../results/pinn_summary_stats.csv", index=False)
print(summary.to_string(index=False))

print("\nSaved: sindy_summary_stats.csv, neural_ode_summary_stats.csv, "
      "pinn_summary_stats.csv in ../results/")
