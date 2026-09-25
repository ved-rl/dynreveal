
import pysindy as ps
import pandas as pd
from systems import SYSTEMS, simulate, corrupt, corrupt_regular, corrupt_block
from scoring import score_model

PATTERNS = {"random": corrupt, "regular": corrupt_regular, "block": corrupt_block}


CONDITIONS = [
    (0.10, 0.50),
    (0.10, 0.25),
    (0.30, 0.05),
]
N_SEEDS = 3
THRESHOLD = 0.10


def build_library(system):
    if system["name"] == "duffing":
        lib_state = ps.PolynomialLibrary(degree=3)
        lib_clock = ps.PolynomialLibrary(degree=1)
        return ps.GeneralizedLibrary(
            [lib_state, lib_clock], inputs_per_library=[[0, 1], [2, 3]]
        )
    return ps.PolynomialLibrary(degree=3)


def run_one(system, noise_frac, obs_frac, pattern_name, corrupt_fn, seed):
    t, x = simulate(system)
    t_obs, x_obs = corrupt_fn(t, x, noise_frac=noise_frac, obs_frac=obs_frac, seed=seed)
    feature_names = [f"x{i+1}" for i in range(system["n_state"])]
    diff_method = ps.SmoothedFiniteDifference() if noise_frac > 0 else ps.FiniteDifference()

    model = ps.SINDy(
        differentiation_method=diff_method,
        feature_library=build_library(system),
        optimizer=ps.STLSQ(threshold=THRESHOLD),
    )
    model.fit(x_obs, t=t_obs, feature_names=feature_names)
    return score_model(model, system["true_terms"]())


def main():
    rows = []
    total = len(SYSTEMS) * len(CONDITIONS) * len(PATTERNS) * N_SEEDS
    done = 0
    for name, system in SYSTEMS.items():
        for noise_frac, obs_frac in CONDITIONS:
            for pattern_name, corrupt_fn in PATTERNS.items():
                for seed in range(N_SEEDS):
                    result = run_one(system, noise_frac, obs_frac, pattern_name, corrupt_fn, seed)
                    row = dict(system=name, noise=noise_frac, obs_frac=obs_frac,
                               pattern=pattern_name, seed=seed,
                               recall=result["recall"], precision=result["precision"])
                    rows.append(row)
                    done += 1
                    if done % 20 == 0:
                        print(f"[{done}/{total}] ...")

    df = pd.DataFrame(rows)
    df.to_csv("../results/sampling_pattern_sensitivity.csv", index=False)

    summary = df.groupby(["system", "noise", "obs_frac", "pattern"]).agg(
        recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
    ).reset_index()
    print("\n" + summary.to_string(index=False))
    print("\nSaved results/sampling_pattern_sensitivity.csv")


if __name__ == "__main__":
    main()
