
import pysindy as ps
from systems import LORENZ, simulate, corrupt
from scoring import score_model

NOISE = 0.10
OBS_FRACS = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.075, 0.05, 0.03]

t, x = simulate(LORENZ)

for obs_frac in OBS_FRACS:
    t_obs, x_obs = corrupt(t, x, noise_frac=NOISE, obs_frac=obs_frac, seed=0)
    model = ps.SINDy(
        differentiation_method=ps.SmoothedFiniteDifference(),
        feature_library=ps.PolynomialLibrary(degree=3),
        optimizer=ps.STLSQ(threshold=0.1),
    )
    model.fit(x_obs, t=t_obs, feature_names=["x1", "x2", "x3"])
    coefs = model.coefficients()
    n_nonzero = (abs(coefs) > 1e-3).sum()
    score = score_model(model, LORENZ["true_terms"]())
    print(f"obs={obs_frac:>5.2f} ({len(t_obs):>4} pts)  "
          f"nonzero={n_nonzero:>3}/60  "
          f"precision={score['precision']:.2f}  recall={score['recall']:.2f}  "
          f"coef_err={score['coef_err']:.2f}")
