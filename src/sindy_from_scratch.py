import numpy as np
from systems import LORENZ, simulate


def poly_library(X, degree=2):
    """Build [1, x1, x2, x3, x1^2, x1x2, ..., ] up to degree."""
    n, d = X.shape
    from itertools import combinations_with_replacement
    terms, names = [], []
    for deg in range(degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            col = np.ones(n)
            for idx in combo:
                col = col * X[:, idx]
            terms.append(col)
            if deg == 0:
                names.append("1")
            else:
                names.append(
                    " ".join(f"x{idx+1}" for idx in combo)
                    .replace("x1 x1", "x1^2").replace("x2 x2", "x2^2")
                    .replace("x3 x3", "x3^2")
                )
    return np.column_stack(terms), names


def finite_diff(t, X):
    """Simple central-difference derivative estimate."""
    dt = t[1] - t[0]
    dX = np.gradient(X, dt, axis=0)
    return dX


def stlsq(Theta, dX, lam=0.1, n_iters=10):
    """Sequential thresholded least squares: fit least squares, zero
    small coefficients, refit on remaining."""
    n_targets = dX.shape[1]
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]  # (n_terms, n_targets)
    for _ in range(n_iters):
        small = np.abs(Xi) < lam
        Xi[small] = 0
        for k in range(n_targets):
            big = ~small[:, k]
            if big.sum() == 0:
                continue
            Xi[big, k] = np.linalg.lstsq(Theta[:, big], dX[:, k], rcond=None)[0]
    return Xi


def print_model(Xi, names, state_names=("dx1", "dx2", "dx3"), tol=1e-6):
    for k, sname in enumerate(state_names):
        terms = [f"{Xi[i,k]:+.3f} {names[i]}" for i in range(len(names))
                  if abs(Xi[i, k]) > tol]
        print(f"  {sname}/dt = " + " ".join(terms))


if __name__ == "__main__":
    # 1. Simulate clean Lorenz data
    t, X = simulate(LORENZ)

    # 2. Build library + estimate derivatives
    Theta, names = poly_library(X, degree=2)
    dX = finite_diff(t, X)

    # 3. Fit via STLSQ
    Xi = stlsq(Theta, dX, lam=0.1)

    print("Recovered model (clean data and dense sampling):")
    print_model(Xi, names)

    print("\nTrue Lorenz equations (sigma=10, rho=28, beta=8/3):")
    print("  dx1/dt = -10.000 x1 +10.000 x2")
    print("  dx2/dt = +28.000 x1 -1.000 x2 -1.000 x1 x3")
    print("  dx3/dt = -2.667 x3 +1.000 x1 x2")
