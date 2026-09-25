
import pysindy as ps
from systems import LORENZ, simulate
from scoring import score_model

t, x = simulate(LORENZ)

model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3))
model.fit(x, t=t, feature_names=["x1", "x2", "x3"])

print("PySINDy recovered model (clean Lorenz data):")
model.print()

true_terms = LORENZ["true_terms"]()
print("\nStructural recovery score vs. ground truth:")
print(score_model(model, true_terms))
