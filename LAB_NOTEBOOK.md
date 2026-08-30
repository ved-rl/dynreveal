# Lab Notebook — dynreveal

 environment set up, systems validated, first working equation-discovery result.

**What I did:**
- Set up repo, venv, git
- Simulated and visually validated Lorenz / Van der Pol / Duffing (see figs/system_validation.png)
- Implemented SINDy from scratch (STLSQ over polynomial library) recovers exact
  Lorenz coefficients from clean data w ~1% error

**Notes**
- Duffing has time-dependence, and none of the 3 methods
  handle this for free; need to decide how each one sees it
  (accept as known limitation?)
- finite-difference derivative estimate is fine for clean data; will need a
  smoothed derivative once noise enters 

**questions for tomorrow:**
- [Research workarounds for time-dependence limitation and smoothed derivatives ]

