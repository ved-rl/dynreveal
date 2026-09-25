
import numpy as np
from systems import VDP, simulate, corrupt
from pinn import train_pinn, score_params

CONDITIONS = [(0.05, 1.00), (0.10, 0.50)]
ITER_LEVELS = [15000, 30000]
N_SEEDS = 5

true_values = dict(mu=VDP["params"]["mu"])

for noise_frac, obs_frac in CONDITIONS:
    for n_iters in ITER_LEVELS:
        errors = []
        for seed in range(N_SEEDS):
            t, x = simulate(VDP)
            t_obs, x_obs = corrupt(t, x, noise_frac=noise_frac, obs_frac=obs_frac, seed=seed)
            net, params = train_pinn(
                t_obs, x_obs, system_name="vanderpol", n_state=2, t_span=VDP["t_span"],
                n_iters=n_iters, verbose_every=0,
            )
            score = score_params(params, true_values)
            errors.append(score["mean_rel_error"])

        print(f"noise={noise_frac} obs={obs_frac} iters={n_iters:5d}  "
              f"errors={[f'{e:.3f}' for e in errors]}  "
              f"mean={np.mean(errors):.3f}+-{np.std(errors):.3f}")
