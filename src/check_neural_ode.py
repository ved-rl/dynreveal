
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from systems import LORENZ, simulate
from neural_ode import train_neural_ode, rollout, evaluate_rollout, save_model

t, x = simulate(LORENZ)

print("Training Neural ODE on clean Lorenz data (window kept within Lorenz's "
      "predictability horizon, ~1 time unit -- this will take a minute or two)...")
func = train_neural_ode(t, x, n_state=3, window_time=0.25, batch_size=32, n_iters=1500)

save_model(func, "../results/neural_ode_lorenz_clean.pt")
print("Saved model weights to results/neural_ode_lorenz_clean.pt "
      "(so a closed terminal doesn't lose this again)")

print("\nRolling out from the true initial condition across the full time span...")
pred = rollout(func, x[0], t)

score = evaluate_rollout(pred, x, short_horizon_frac=0.05)  # ~1 time unit of 20 -- Lorenz's approx. predictability horizon
print("\nEvaluation (short_rmse measured within the predictability horizon, "
      "not the full chaotic rollout):", score)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels = ["x1", "x2", "x3"]
for i, ax in enumerate(axes):
    ax.plot(t, x[:, i], label="true", lw=1)
    ax.plot(t, pred[:, i], label="predicted", lw=1, ls="--")
    ax.set_title(labels[i])
    ax.set_xlabel("t")
axes[0].legend()
plt.tight_layout()
plt.savefig("../figs/neural_ode_lorenz_clean.png", dpi=140)
print("Saved figs/neural_ode_lorenz_clean.png")
