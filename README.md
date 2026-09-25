# dynreveal

A research project I've been working on since April 2026: can machine learning methods recover the correct equations of a dynamical system when the data is noisy or incomplete and does that depend on which method you use?

I compared three approaches, SINDy (sparse regression over a library of candidate terms), Neural ODEs (a neural network trained directly on the vector field), and PINNs (physics informed neural networks, which assume the equation form is known and only recover the parameters). I run them on three systems, Lorenz, Van der Pol, and Duffing, across different levels of noise and missing data, checking not just how well they predict but whether they actually recover the right equation or parameters.

## Status

Synthetic experiments are done for all three methods, across a noise and sparsity grid, with 5 random seeds per condition. I also ran a set of robustness checks on top of the main results: SINDy's sensitivity to its sparsity threshold and to how the missing data is patterned (random, regular, or a single missing block), and a targeted follow up test that separated training window length from training budget for both Neural ODE and PINN, since an early result had those two things confounded. I validated the core SINDy finding on Hudson Bay Company lynx-hare pelt records, 1900-1920 as well.

Each method turned out to fail in a different way once the data gets bad. SINDy keeps finding the right terms but starts adding extra wrong ones as data gets sparse. Neural ODE gets steadily worse with noise, and what first looked like a chaos-specific breakdown on Lorenz turned out to mostly be a training budget issue once I tested it properly. PINN barely notices noise but falls apart on sparse chaotic data in a way that depends heavily on the random seed.

A full paper draft is written and currently being revised with the goal of an preprint first and a TMLR submission after.

## Repo layout

```
src/
  systems.py                        ground-truth ODEs, simulation, noise and sparsity corruption
  scoring.py                        grades a SINDy model on structural recovery (precision, recall)
  validate_systems.py               sanity-check plots of the three systems
  sindy_from_scratch.py             minimal handwritten SINDy, verified on clean Lorenz data
  check_pysindy.py                  confirms the real PySINDy library agrees with the handwritten version
  sindy_grid.py, plot_failure_boundary.py       main SINDy noise/sparsity grid and plot
  debug_duffing.py, check_sparsity_bloat.py, find_cliff.py, find_cliff_v2.py, check_ensemble_sindy.py
                                     the debugging trail behind the SINDy results (Duffing forcing term
                                     fix, the non-sparse collapse under sparsity, ruling out normalization
                                     and Ensemble-SINDy as fixes)
  threshold_sensitivity.py, sampling_pattern_sensitivity.py
                                     robustness checks on SINDy's threshold and missing-data pattern
  neural_ode.py, check_neural_ode.py, neural_ode_grid.py, plot_neural_ode_grid.py
                                     Neural ODE model, clean-data check, main grid, and plot
  node_confound_test.py             separates training window length from training budget on Lorenz
  pinn.py, check_pinn.py, check_pinn_vdp.py, pinn_grid.py, pinn_extra_seeds.py,
  pinn_vdp_iteration_check.py, plot_pinn_grid.py
                                     PINN model, per-system clean-data checks, main grid, extra seeds,
                                     a targeted iteration-budget check on Van der Pol, and the plot
  aggregate_results.py              turns the raw per-seed CSVs into mean +/- std summary tables
  real_world_sindy.py               SINDy on the real lynx-hare dataset
figs/                                generated plots
results/                             raw and summary CSVs from every experiment
data/                                lynx_hare.csv, the real-world validation dataset
paper/                               paper.tex, the current draft
LAB_NOTEBOOK.md                      running research log
SYNTHESIS.md                         cross-method comparison, the core of the paper's results section
requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Why this project

Most examples you see for these methods use perfect simulated data, which isn't how real data works. I wanted to test where these methods start to break down under realistic conditions and see if there's a simple way to make them hold up longer without building something overly complicated. Along the way this turned into as much a project about doing research carefully as it is about the specific methods themselves.

## License

Haven't decided yet.
