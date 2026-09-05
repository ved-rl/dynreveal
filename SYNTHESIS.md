# Cross-method synthesis

## The core finding

Each method fails differently under degraded data, but **chaos (Lorenz)
consistently amplifies fragility across all three**, regardless of which
paradigm the method belongs to. The specific failure signature differs
by method:

| Method | Primary failure driver | Character of failure | Representative evidence |
|---|---|---|---|
| **SINDy** | Sparsity >> noise | Gradual, monotonic precision collapse to a low floor (~0.15); recall stays robust throughout | Precision 0.42 -> 0.14 as observations drop from 50% to 3% at fixed 10% noise. Model doesn't miss real terms, it adds spurious ones. Ensemble-SINDy (published robust-SINDy baseline) did not fix this. |
| **Neural ODE** | Noise (non-chaotic systems); training-budget-limited on the chaotic one | Clean, monotonic degradation with noise on Van der Pol/Duffing (0.07 -> 0.80 and 0.16 -> 0.52 respectively, full obs). Lorenz result dominated by a predictability-horizon / fixed-point-collapse artifact rather than a clean noise trend. | Van der Pol short-horizon RMSE climbs steadily 0.07->0.80 as noise 0%->50% at full observation. |
| **PINN** | Sparsity, sharply, and specific to the chaotic system | Near-perfect (<2% parameter error) under noise up to 20% and observations down to 50% -- then a sharp cliff to 20-60% error once observations drop to 25%, only on Lorenz. Duffing/Van der Pol stay accurate at the same sparsity level. | Lorenz mean relative parameter error: ~0.01 at 50% obs -> ~0.43 at 25% obs, flat across noise levels once triggered (sparsity failure fully dominates). |

## Why this is the paper's central contribution

Not "we benchmarked three methods and here are three tables" -- the
throughline is that **the same underlying vulnerability (chaos +
sparse/noisy data) manifests as three completely different failure
signatures depending on the method's mechanism**:
- SINDy estimates derivatives pointwise from the raw data -- sparsity
  directly starves that estimate of reliable information, hence its
  gradual precision collapse.
- Neural ODE's failure ties to the predictability horizon -- training
  windows that stretch past ~1 Lyapunov time push it toward hedging
  behavior (regression to the mean/fixed point) rather than committing
  to trackable dynamics.
- PINN's physics constraint, enforced at dense collocation points
  spanning the whole domain, makes it remarkably robust to noise (the
  constraint doesn't degrade with noisy point-data), but that same
  mechanism can't compensate when there's too little real data to
  anchor the collocation constraint to reality in a chaotic regime --
  hence the sharp, sparsity-specific cliff rather than a gradual decline.

## Open items / honest limitations
- Van der Pol's PINN results were too noisy (seed-to-seed variance) to
  draw a clean trend from with only 3 seeds -- reported as an observed
  limitation, not resolved further given time constraints.
- Neural ODE's Lorenz results are confounded by training-budget and
  predictability-horizon effects that weren't fully separated from a
  genuine noise/sparsity trend.
- All grids use 3 systems and a bounded noise/sparsity range (0-20% or
  0-50% depending on method) -- not exhaustive, but consistent across
  methods where directly compared.
