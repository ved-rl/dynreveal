
import numpy as np
import pysindy as ps
from systems import LORENZ, simulate, corrupt
from scoring import score_model

NOISE = 0.10
OBS_FRACS = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.075, 0.05, 0.03]
N_SEEDS = 5

t, x = simulate(LORENZ)

for obs_frac in OBS_FRACS:
    precisions, recalls, nonzeros = [], [], []
    for seed in range(N_SEEDS):
        t_obs, x_obs = corrupt(t, x, noise_frac=NOISE, obs_frac=obs_frac, seed=seed)
        model = ps.SINDy(
            differentiation_method=ps.SmoothedFiniteDifference(),
            feature_library=ps.PolynomialLibrary(degree=3),
            optimizer=ps.STLSQ(threshold=0.1),
        )
        model.fit(x_obs, t=t_obs, feature_names=["x1", "x2", "x3"])
        coefs = model.coefficients()
        nonzeros.append((abs(coefs) > 1e-3).sum())
        score = score_model(model, LORENZ["true_terms"]())
        precisions.append(score["precision"])
        recalls.append(score["recall"])

    print(f"obs={obs_frac:>5.2f}  "
          f"precision={np.mean(precisions):.2f}+-{np.std(precisions):.2f}  "
          f"recall={np.mean(recalls):.2f}+-{np.std(recalls):.2f}  "
          f"nonzero={np.mean(nonzeros):>4.1f}/60")
