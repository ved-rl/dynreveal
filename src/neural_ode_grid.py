
import time
import numpy as np
import pandas as pd
from systems import SYSTEMS, GRID, simulate, corrupt_regular
from neural_ode import train_neural_ode, rollout, evaluate_rollout

N_SEEDS = 5
N_ITERS = 1000


def run_one(system, noise_frac, obs_frac, seed):
    t, x = simulate(system)
    t_obs, x_obs = corrupt_regular(t, x, noise_frac=noise_frac, obs_frac=obs_frac, seed=seed)

    start = time.time()
    func = train_neural_ode(
        t_obs, x_obs, n_state=system["n_state"],
        window_time=0.25, batch_size=16, n_iters=N_ITERS, verbose_every=0,
    )
    train_seconds = time.time() - start

    pred = rollout(func, x[0], t)
    score = evaluate_rollout(pred, x, short_horizon_frac=0.05)
    score["train_seconds"] = train_seconds
    return score


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
                      f"seed={seed} -> short_rmse={result['short_rmse']:.2f} "
                      f"blew_up={result['blew_up']} ({result['train_seconds']:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv("../results/neural_ode_grid_results.csv", index=False)
    print("\nSaved results/neural_ode_grid_results.csv")


if __name__ == "__main__":
    main()
