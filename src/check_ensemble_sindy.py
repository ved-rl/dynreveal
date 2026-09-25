
import numpy as np
import pysindy as ps
from systems import LORENZ, simulate, corrupt
from scoring import score_model

NOISE = 0.10
OBS_FRACS = [0.50, 0.25, 0.15, 0.10, 0.05]
N_SEEDS = 5

t, x = simulate(LORENZ)

for obs_frac in OBS_FRACS:
    precisions, recalls = [], []
    for seed in range(N_SEEDS):
        t_obs, x_obs = corrupt(t, x, noise_frac=NOISE, obs_frac=obs_frac, seed=seed)
        base_opt = ps.STLSQ(threshold=0.1)
        ensemble_opt = ps.EnsembleOptimizer(base_opt, bagging=True, n_models=20)
        model = ps.SINDy(
            differentiation_method=ps.SmoothedFiniteDifference(),
            feature_library=ps.PolynomialLibrary(degree=3),
            optimizer=ensemble_opt,
        )
        model.fit(x_obs, t=t_obs, feature_names=["x1", "x2", "x3"])
        score = score_model(model, LORENZ["true_terms"]())
        precisions.append(score["precision"])
        recalls.append(score["recall"])

    print(f"obs={obs_frac:>5.2f}  "
          f"precision={np.mean(precisions):.2f}+-{np.std(precisions):.2f}  "
          f"recall={np.mean(recalls):.2f}+-{np.std(recalls):.2f}")
