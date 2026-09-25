
import pysindy as ps
from systems import LORENZ, simulate, corrupt
from scoring import score_model

t, x = simulate(LORENZ)
t_obs, x_obs = corrupt(t, x, noise_frac=0.3, obs_frac=0.05, seed=0)

model = ps.SINDy(
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_library=ps.PolynomialLibrary(degree=3),
    optimizer=ps.STLSQ(threshold=0.1, normalize_columns=True),
)
model.fit(x_obs, t=t_obs, feature_names=["x1", "x2", "x3"])

print(f"Observed points used: {len(t_obs)} (of {len(t)} total)")
print("\nRecovered model (30% noise, 5% obs):")
model.print()

coefs = model.coefficients()
n_total_candidates = coefs.shape[1]
n_nonzero = (abs(coefs) > 1e-3).sum()
print(f"\nNonzero terms kept: {n_nonzero} out of {n_total_candidates} candidates "
      f"per equation x {coefs.shape[0]} equations")

print("\nScore:", score_model(model, LORENZ["true_terms"]()))
