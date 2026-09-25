import pysindy as ps
import pandas as pd
from systems import SYSTEMS, GRID, simulate, corrupt
from scoring import score_model

N_SEEDS = 5


def build_library(system):
    if system["name"] == "duffing":
        lib_state = ps.PolynomialLibrary(degree=3)
        lib_clock = ps.PolynomialLibrary(degree=1)
        return ps.GeneralizedLibrary(
            [lib_state, lib_clock], inputs_per_library=[[0, 1], [2, 3]]
        )
    return ps.PolynomialLibrary(degree=3)


def run_one(system, noise_frac, obs_frac, seed):
    t, x = simulate(system)
    t_obs, x_obs = corrupt(t, x, noise_frac=noise_frac, obs_frac=obs_frac, seed=seed)

    feature_names = [f"x{i+1}" for i in range(system["n_state"])]
    diff_method = ps.SmoothedFiniteDifference() if noise_frac > 0 else ps.FiniteDifference()

    model = ps.SINDy(
        differentiation_method=diff_method,
        feature_library=build_library(system),
        optimizer=ps.STLSQ(threshold=0.1),
    )
    try:
        model.fit(x_obs, t=t_obs, feature_names=feature_names)
        return score_model(model, system["true_terms"]())
    except Exception as e:
        return dict(precision=0.0, recall=0.0, coef_err=float("nan"), error=str(e))


def main():
    rows = []
    total = len(SYSTEMS) * len(GRID) * N_SEEDS
    done = 0
    for name, system in SYSTEMS.items():
        for noise_frac, obs_frac in GRID:
            for seed in range(N_SEEDS):
                result = run_one(system, noise_frac, obs_frac, seed)
                row = dict(system=name, noise=noise_frac, obs_frac=obs_frac, seed=seed)
                row.update(result)
                rows.append(row)
                done += 1
                print(f"[{done}/{total}] {name} noise={noise_frac} obs={obs_frac} "
                      f"seed={seed} -> recall={result.get('recall'):.2f}")

    df = pd.DataFrame(rows)
    df.to_csv("../results/sindy_grid_results.csv", index=False)
    print("\nSaved results/sindy_grid_results.csv")


if __name__ == "__main__":
    main()
