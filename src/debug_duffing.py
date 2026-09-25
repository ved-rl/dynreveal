
import pysindy as ps
from systems import DUFFING, simulate
from scoring import score_model

t, x = simulate(DUFFING)

lib_state = ps.PolynomialLibrary(degree=3)
lib_clock = ps.PolynomialLibrary(degree=1)
library = ps.GeneralizedLibrary([lib_state, lib_clock], inputs_per_library=[[0, 1], [2, 3]])

model = ps.SINDy(feature_library=library)
model.fit(x, t=t, feature_names=["x1", "x2", "x3", "x4"])

print("Recovered model (clean, 100% observed Duffing data):")
model.print()

true_terms = DUFFING["true_terms"]()
print("\nTrue terms we're checking against:", true_terms)
print("\nScore:", score_model(model, true_terms))
