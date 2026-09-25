
import time
import pandas as pd
from systems import SYSTEMS, simulate, corrupt
from pinn import train_pinn, score_params

GRID = [
    (0.00, 1.00),
    (0.01, 1.00),
    (0.05, 1.00),
    (0.10, 1.00),
    (0.20, 1.00),
    (0.05, 0.50),
    (0.10, 0.50),
    (0.10, 0.25),
    (0.20, 0.25),
]
N_SEEDS = 3
N_ITERS = 15000

DOMAIN_LIMIT = {"lorenz": 3.0, "vanderpol": None, "duffing": None}


def get_true_values(system):
    name, p = system["name"], system["params"]
    if name == "lorenz":
        return dict(sigma=p["sigma"], rho=p["rho"], beta=p["beta"])
    if name == "vanderpol":
        return dict(mu=p["mu"])
    if name == "duffing":
        return dict(delta=p["delta"], alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"])


def run_one(system, noise_frac, obs_frac, seed):
    t, x = simulate(system)

    limit = DOMAIN_LIMIT[system["name"]]
    if limit is not None:
        n_short = int(limit / system["dt"])
        t, x = t[:n_short], x[:n_short]

    t_obs, x_obs = corrupt(t, x, noise_frac=noise_frac, obs_frac=obs_frac, seed=seed)
    t_span = (t[0], t[-1])

    kwargs = dict(n_iters=N_ITERS, verbose_every=0)
    if system["name"] == "duffing":
        kwargs["omega"] = system["params"]["omega"]

    start = time.time()
    net, params = train_pinn(
        t_obs, x_obs, system_name=system["name"], n_state=system["n_state"],
        t_span=t_span, **kwargs,
    )
    train_seconds = time.time() - start

    score = score_params(params, get_true_values(system))
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
                row = dict(system=name, noise=noise_frac, obs_frac=obs_frac, seed=seed,
                           mean_rel_error=result["mean_rel_error"],
                           train_seconds=result["train_seconds"])
                rows.append(row)
                done += 1
                print(f"[{done}/{total}] {name} noise={noise_frac} obs={obs_frac} "
                      f"seed={seed} -> mean_rel_error={result['mean_rel_error']:.3f} "
                      f"({result['train_seconds']:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv("../results/pinn_grid_results.csv", index=False)
    print("\nSaved results/pinn_grid_results.csv")


if __name__ == "__main__":
    main()
