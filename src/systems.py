import numpy as np
from scipy.integrate import solve_ivp


LORENZ_PARAMS = dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0)


def lorenz_rhs(t, x, params=LORENZ_PARAMS):
    x1, x2, x3 = x
    sigma, rho, beta = params["sigma"], params["rho"], params["beta"]
    dx1 = sigma * (x2 - x1)
    dx2 = x1 * (rho - x3) - x2
    dx3 = x1 * x2 - beta * x3
    return [dx1, dx2, dx3]


def lorenz_true_terms(params=LORENZ_PARAMS):
    sigma, rho, beta = params["sigma"], params["rho"], params["beta"]
    return {
        "dx1": {"x1": -sigma, "x2": sigma},
        "dx2": {"x1": rho, "x2": -1.0, "x1 x3": -1.0},
        "dx3": {"x3": -beta, "x1 x2": 1.0},
    }


LORENZ = dict(
    name="lorenz", rhs=lorenz_rhs, params=LORENZ_PARAMS, true_terms=lorenz_true_terms,
    x0=[1.0, 1.0, 1.0], t_span=(0.0, 20.0), dt=0.01, n_state=3,
)

VDP_PARAMS = dict(mu=2.0)


def vdp_rhs(t, x, params=VDP_PARAMS):
    x1, x2 = x
    mu = params["mu"]
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    return [dx1, dx2]


def vdp_true_terms(params=VDP_PARAMS):
    mu = params["mu"]
    return {
        "dx1": {"x2": 1.0},
        "dx2": {"x1": -1.0, "x2": mu, "x1^2 x2": -mu},
    }


VDP = dict(
    name="vanderpol", rhs=vdp_rhs, params=VDP_PARAMS, true_terms=vdp_true_terms,
    x0=[2.0, 0.0], t_span=(0.0, 30.0), dt=0.01, n_state=2,
)

DUFFING_PARAMS = dict(delta=0.3, alpha=-1.0, beta=1.0, gamma=0.37, omega=1.2)


def duffing_rhs(t, x, params=DUFFING_PARAMS):
    x1, x2, c, s = x
    delta, alpha, beta, gamma, omega = (
        params["delta"], params["alpha"], params["beta"],
        params["gamma"], params["omega"],
    )
    dx1 = x2
    dx2 = -delta * x2 - alpha * x1 - beta * x1**3 + gamma * c
    dc = -omega * s
    ds = omega * c
    return [dx1, dx2, dc, ds]


def duffing_true_terms(params=DUFFING_PARAMS):
    delta, alpha, beta, gamma, omega = (
        params["delta"], params["alpha"], params["beta"],
        params["gamma"], params["omega"],
    )
    return {
        "dx1": {"x2": 1.0},
        "dx2": {"x2": -delta, "x1": -alpha, "x1^3": -beta, "x3": gamma},
        "dx3": {"x4": -omega},
        "dx4": {"x3": omega},
    }


DUFFING = dict(
    name="duffing", rhs=duffing_rhs, params=DUFFING_PARAMS, true_terms=duffing_true_terms,
    x0=[1.0, 0.0, 1.0, 0.0], t_span=(0.0, 40.0), dt=0.01, n_state=4,
)


SYSTEMS = {"lorenz": LORENZ, "vanderpol": VDP, "duffing": DUFFING}

def simulate(system, t_span=None, dt=None, x0=None, rtol=1e-10, atol=1e-12):
    t_span = t_span or system["t_span"]
    dt = dt or system["dt"]
    x0 = x0 or system["x0"]
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        system["rhs"], t_span, x0, t_eval=t_eval,
        args=(system["params"],), rtol=rtol, atol=atol, method="RK45",
    )
    return sol.t, sol.y.T


def corrupt(t, x, noise_frac=0.0, obs_frac=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x_noisy = x.copy()
    if noise_frac > 0:
        std = x.std(axis=0, keepdims=True)
        x_noisy = x + rng.normal(0, noise_frac * std, size=x.shape)

    n = len(t)
    if obs_frac < 1.0:
        n_keep = max(int(round(obs_frac * n)), 5)
        idx = np.sort(rng.choice(n, size=n_keep, replace=False))
    else:
        idx = np.arange(n)

    return t[idx], x_noisy[idx]


def corrupt_regular(t, x, noise_frac=0.0, obs_frac=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x_noisy = x.copy()
    if noise_frac > 0:
        std = x.std(axis=0, keepdims=True)
        x_noisy = x + rng.normal(0, noise_frac * std, size=x.shape)

    n = len(t)
    if obs_frac < 1.0:
        n_keep = max(int(round(obs_frac * n)), 5)
        idx = np.linspace(0, n - 1, n_keep).astype(int)
    else:
        idx = np.arange(n)

    return t[idx], x_noisy[idx]


def corrupt_block(t, x, noise_frac=0.0, obs_frac=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x_noisy = x.copy()
    if noise_frac > 0:
        std = x.std(axis=0, keepdims=True)
        x_noisy = x + rng.normal(0, noise_frac * std, size=x.shape)

    n = len(t)
    if obs_frac < 1.0:
        n_keep = max(int(round(obs_frac * n)), 5)
        n_remove = n - n_keep
        start = rng.integers(0, n - n_remove + 1)
        idx = np.concatenate([np.arange(0, start), np.arange(start + n_remove, n)])
    else:
        idx = np.arange(n)

    return t[idx], x_noisy[idx]

GRID = [
    (0.00, 1.00), (0.05, 1.00), (0.10, 1.00), (0.20, 1.00), (0.30, 1.00), (0.40, 1.00), (0.50, 1.00),
    (0.05, 0.50), (0.10, 0.50), (0.20, 0.50), (0.30, 0.50),
    (0.05, 0.25), (0.10, 0.25), (0.20, 0.25), (0.30, 0.25),
    (0.10, 0.10), (0.20, 0.10), (0.30, 0.10),
    (0.20, 0.05), (0.30, 0.05),
]
