import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

df = pd.read_csv("../data/lynx_hare.csv")
t = (df["year"] - df["year"].iloc[0]).values.astype(float)
x = df[["hares", "lynx"]].values.astype(float)

print(f"Loaded {len(t)} real annual observations, {t[0]:.0f} to {t[0]+t[-1]:.0f} years span.")

model = ps.SINDy(
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_library=ps.PolynomialLibrary(degree=2, include_bias=False),
    optimizer=ps.STLSQ(threshold=0.01),
)
model.fit(x, t=t, feature_names=["hares", "lynx"])

print("\nSINDy-recovered model from real data:")
model.print()

print("\nLiterature reference (Bayesian Spline Learning, 2022):")
print("  d(hares)/dt = 0.4807 hares - 0.0248 hares*lynx")
print("  d(lynx)/dt  = -0.9272 lynx + 0.0276 hares*lynx")

def rhs(tt, y):
    return model.predict(y[np.newaxis, :])[0]

def blowup(tt, y):
    return 300 - np.max(np.abs(y))
blowup.terminal = True
blowup.direction = -1

sol = solve_ivp(rhs, (t[0], t[-1]), x[0], method="RK45", rtol=1e-6, atol=1e-6,
                 events=blowup, dense_output=True)

if sol.status == 1:
    print(f"Note: recovered model diverges -- exceeded bound at t={sol.t[-1]:.1f} "
          f"(year {1900 + sol.t[-1]:.0f}). Plotting only up to that point.")

t_dense = np.linspace(sol.t[0], sol.t[-1], 200)
x_sim = sol.sol(t_dense).T

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
labels = ["Hares (x1000 pelts)", "Lynx (x1000 pelts)"]
for i, ax in enumerate(axes):
    ax.scatter(t + 1900, x[:, i], label="real data", color="black", zorder=5, s=20)
    ax.plot(t_dense + 1900, x_sim[:, i], label="SINDy reconstruction", color="C1")
    ax.set_title(labels[i])
    ax.set_xlabel("year")
axes[0].legend()
plt.tight_layout()
plt.savefig("../figs/lynx_hare_sindy.png", dpi=140)
print("\nSaved figs/lynx_hare_sindy.png")
