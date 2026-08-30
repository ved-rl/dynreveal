# dynreveal

**Can data-driven and physics-informed ML methods reliably discover the governing
equations of a dynamical system when observations are sparse and noisy — and does
that reliability differ systematically across method classes?**

This project benchmarks three families of equation/dynamics-discovery methods —
**SINDy** (sparse regression over a candidate function library), **Neural ODEs**,
and **Physics-Informed Neural Networks (PINNs)** — across a controlled grid of
observation noise and sampling sparsity, on three canonical nonlinear systems
(Lorenz, Van der Pol, Duffing), measuring not just prediction accuracy but whether
the *correct governing equation structure* is recovered.

## Status

🚧 Early / in progress. Currently validating ground-truth simulations and a
from-scratch SINDy baseline before running the full experiment grid.

## Repo layout

```
src/
  systems.py             # ground-truth ODEs, simulation, noise/sparsity corruption
  validate_systems.py    # sanity-check plots of the three systems
  sindy_from_scratch.py  # minimal STLSQ SINDy, verified on clean Lorenz data
figs/                     # generated plots
results/                  # experiment outputs (grid runs, metrics)
data/                     # real-world validation dataset (TBD)
LAB_NOTEBOOK.md            # running research log
requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Why this project

Real measurements are rarely clean or complete — sensors are noisy, expensive,
missing, or irregularly sampled. Understanding *when* equation-discovery methods
can still be trusted under those conditions, and whether that failure point can be
pushed back with simple, honest preprocessing, is directly relevant to how these
methods get used on real scientific and engineering data.

## License

TBD
