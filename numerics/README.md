# Numerical experiments

All commands run from the repository root with Python 3.11+ and `numpy`,
`scipy`, `pandas`, `matplotlib`, and `numba`.

## Section 5: simulated hedging

- `original_market_stability.py`: common-path simulation and the fixed `R=20`
  strategy and gain convergence at two time grids.
- `plot_hedging_convergence_paper.py`: the convergence figure from the saved CSV.
- `terminal_mse_grid_check.py`: the four-strategy comparison on paired 256- and
  512-step grids, using specified initial capitals and paired standard errors.
- `test_original_market_stability.py`: kernel-integral, Riccati normalization,
  Fourier quadrature, and history-reconstruction checks.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.12 numerics/original_market_stability.py
python3.12 numerics/plot_hedging_convergence_paper.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.12 numerics/terminal_mse_grid_check.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.12 numerics/test_original_market_stability.py
```

The convergence run uses 5,000 paths; the terminal-loss run uses 50,000 paths.
The simulation scripts accept `--help` and write CSV/JSON files under their
corresponding subdirectories of `numerics/output/`. The convergence figure is
`arxiv/figures/hedging_convergence.pdf`. Its plotting command uses stored
results and does not rerun the simulation.

## Section 6: public SPX calibration

`calibrate_public_spx.py` independently calibrates the rough Hawkes--Heston
model and its `nu=0` rough Heston control to four public SPX snapshots, using
the relative-IV objective in Bondi et al. (2024), Section 6.1, equation (42).
Both models use the same observations; an entire expiry and selected strikes
are held out. Optimization and quote-derived initial guesses use only the
training set. These are cross-sectional pricing checks.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.12 numerics/calibrate_public_spx.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.12 numerics/test_public_spx_calibration.py
python3.12 numerics/summarize_public_spx.py
python3.12 numerics/plot_public_spx_paper.py
```

Raw snapshots and their pinned upstream URLs/SHA-256 checksums are in
`numerics/data/public_spx/`. Date-specific JSON files retain optimization
logs, fitted parameters, and quadrature-refinement checks.
`test_public_spx_calibration.py` checks the Black and classical Heston limits,
normalization, quadrature refinement, and an independent fractional solver.
It also verifies sufficient finite-horizon second-moment conditions for the
fitted models, supporting the paper's square-integrability assumption.
`summarize_public_spx.py` generates the parameter/metric tables, combined
predictions, and midpoint-arbitrage diagnostics in
`numerics/output/public_spx_calibration/`.
`plot_public_spx_paper.py` generates `arxiv/figures/public_spx_smiles.pdf`
from the saved predictions.
