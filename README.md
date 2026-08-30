# dynreveal

A small research project I'm working through: can machine learning methods actually recover the correct equations of a dynamical system when the data you're given is noisy or incomplete and does that depend on which method you use?

I'm comparing three approaches. SINDy (sparse regression over a library of candidate terms), Neural ODEs, and Physics Informed Neural Networks (PINNs). I'm running them on three systems, Lorenz, Van der Pol, and Duffing, across different levels of noise and missing data, checking whether they actually recover the right equation.

## Status

This is my first research project, so I'm building it slowly and trying to actually understand each piece instead of just calling libraries blind. Right now I have the three systems simulated and checked against known behavior, plus a basic SINDy implementation I wrote by hand that correctly recovers the Lorenz equations from clean data. Next up is seeing what happens once noise and missing data get added in.

## Repo layout

```
src/
  systems.py             ground-truth ODEs, simulation, noise/sparsity corruption
  validate_systems.py    sanity-check plots of the three systems
  sindy_from_scratch.py  minimal STLSQ SINDy, verified on clean Lorenz data
figs/                     generated plots
results/                  experiment outputs (grid runs, metrics)
data/                     real-world validation dataset (TBD)
LAB_NOTEBOOK.md            running research log
requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Why this project

Most examples you see for these methods use clean, perfect simulated data, which isn't really how real measurements work. I wanted to test where these methods start to break down under realistic conditions and see if there's a simple way to make them hold up longer without building something overly complicated.

## License

Haven't decided yet.
