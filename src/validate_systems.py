"""
Sanity-check script: simulate all three systems, plot clean trajectories
and one example noisy/sparse corruption, so you can visually confirm the
dynamics look right (chaotic butterfly, limit cycle, forced oscillation)
before building any discovery method on top of them.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from systems import SYSTEMS, simulate, corrupt

fig = plt.figure(figsize=(15, 10))

for i, (name, sysdef) in enumerate(SYSTEMS.items()):
    t, x = simulate(sysdef)
    t_obs, x_obs = corrupt(t, x, noise_frac=0.05, obs_frac=0.25, seed=0)

    if sysdef["n_state"] == 3:
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.plot(x[:, 0], x[:, 1], x[:, 2], lw=0.5)
        ax.set_title(f"{name} (clean, phase space)")
    else:
        ax = fig.add_subplot(2, 3, i + 1)
        ax.plot(x[:, 0], x[:, 1], lw=0.5)
        ax.set_title(f"{name} (clean, phase space)")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")

    ax2 = fig.add_subplot(2, 3, i + 4)
    ax2.plot(t, x[:, 0], lw=1, label="clean x1", color="C0")
    ax2.scatter(t_obs, x_obs[:, 0], s=8, color="C1",
                label="5% noise, 25% obs", zorder=5)
    ax2.set_title(f"{name}: clean vs. corrupted (x1)")
    ax2.set_xlabel("t")
    ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig("../figs/system_validation.png", dpi=140)
print("Saved figs/system_validation.png")

# Quick numeric sanity checks
import numpy as np
for name, sysdef in SYSTEMS.items():
    t, x = simulate(sysdef)
    print(f"{name}: T={len(t)} pts, x range = "
          f"{np.round(x.min(0), 2)} to {np.round(x.max(0), 2)}")
