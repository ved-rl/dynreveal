# Lab Notebook — dynreveal

## Day 1 — [fill in today's date]

**Goal:** environment set up, systems validated, first working equation-discovery result.

**What I did:**
- Set up repo, venv, git
- Simulated & visually validated Lorenz / Van der Pol / Duffing (see figs/system_validation.png)
- Implemented SINDy from scratch (STLSQ over polynomial library) — recovers exact
  Lorenz coefficients from clean, densely-sampled data (~1% error)

**Decisions / things to remember:**
- Duffing has explicit time-dependence (forcing term) — none of the 3 methods
  handle this for free; need to decide (Day 2) how each one sees it
  (augmented state? raw time input? accept as known limitation?)
- finite-difference derivative estimate is fine for clean data; will need a
  smoothed/spline-based derivative once noise enters (Day 2-3 problem)

**Open questions for tomorrow:**
- [ ]
