
import time
import numpy as np
from systems import LORENZ, simulate
from neural_ode import train_neural_ode, rollout

WINDOWS = {"short (0.25)": 0.25, "long (1.00)": 1.00}
ITERS = {"modest (1500)": 1500, "high (8000)": 8000}
N_SEEDS = 2

t, x = simulate(LORENZ)
true_late_std = x[len(x) * 3 // 4:, 0].std()
print(f"True data's late-time (last quarter) std, x1: {true_late_std:.2f}\n")

for window_name, window_time in WINDOWS.items():
    for iters_name, n_iters in ITERS.items():
        rmses, late_stds, times = [], [], []
        for seed in range(N_SEEDS):
            np.random.seed(seed)
            start = time.time()
            func = train_neural_ode(t, x, n_state=3, window_time=window_time,
                                      batch_size=32, n_iters=n_iters, verbose_every=0)
            elapsed = time.time() - start
            pred = rollout(func, x[0], t)

            h = max(int(len(t) * 0.05), 5)
            short_rmse = np.sqrt(np.mean((pred[:h] - x[:h]) ** 2))
            late_std = pred[len(pred) * 3 // 4:, 0].std()

            rmses.append(short_rmse)
            late_stds.append(late_std)
            times.append(elapsed)

        print(f"window={window_name:14s} iters={iters_name:14s} "
              f"short_rmse={np.mean(rmses):.2f}+-{np.std(rmses):.2f}  "
              f"late_time_std={np.mean(late_stds):.2f}+-{np.std(late_stds):.2f}  "
              f"(true={true_late_std:.2f})  "
              f"avg_time={np.mean(times):.0f}s")
