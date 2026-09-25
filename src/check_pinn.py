
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from systems import LORENZ, simulate
from pinn import train_pinn, predict, score_params

t, x = simulate(LORENZ)

t, x = t[:n_short], x[:n_short]
t_span = (t[0], t[-1])

true_values = dict(sigma=LORENZ["params"]["sigma"], rho=LORENZ["params"]["rho"],
                    beta=LORENZ["params"]["beta"])

print(f"Training PINN on clean Lorenz data, first {t_span[1]:.1f} time units only "
      "(starting from deliberately wrong parameter guesses)...")
net, params = train_pinn(
    t, x, system_name="lorenz", n_state=3, t_span=t_span,
    n_iters=15000, verbose_every=1000,
)

score = score_params(params, true_values)
print("\nRecovered parameters:", score["current"])
print("True parameters:     ", score["true"])
print("Relative errors:     ", score["rel_errors"])
print("Mean relative error: ", score["mean_rel_error"])

pred = predict(net, t)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels = ["x1", "x2", "x3"]
for i, ax in enumerate(axes):
    ax.plot(t, x[:, i], label="true", lw=1)
    ax.plot(t, pred[:, i], label="predicted", lw=1, ls="--")
    ax.set_title(labels[i])
    ax.set_xlabel("t")
axes[0].legend()
plt.tight_layout()
plt.savefig("../figs/pinn_lorenz_clean.png", dpi=140)
print("Saved figs/pinn_lorenz_clean.png")
