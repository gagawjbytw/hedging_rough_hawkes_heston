# Numerical code

The code requires Python 3.10 or later with `numpy`, `scipy`, and
`matplotlib`. Run commands from the repository root.

## Run the experiment

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/original_market_stability.py
```

The default run uses 5,000 paths, a 512-step fine grid, a 256-step coarse
grid, maturity `0.25`, and seed `20260907`. It writes

- `numerics/output/original_market_stability/convergence.csv`;
- `numerics/output/original_market_stability/summary.json`;
- `figures/original_market_stability.pdf`.

To see all command-line options:

```bash
python3 numerics/original_market_stability.py --help
```

For example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/original_market_stability.py \
  --paths 10000 --steps 1024 --seed 20260909
```

## Run the checks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/test_original_market_stability.py
```

## Terminal-loss and paired grid check

The terminal-loss experiment compares each Fourier policy with an unhedged
position and a Black--Scholes delta hedge on the same original-market paths:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/hedging_performance_experiment.py
```

Its default run uses 50,000 paths and 512 steps, and writes
`numerics/output/hedging_performance_experiment/summary.json` and
`terminal_mse.csv` in the same directory.

The paired grid check evaluates the final regularized policy, its unshifted
reference, and both benchmarks at 256 and 512 time steps. It aggregates the
fine Brownian normals in adjacent pairs and reuses the Poisson thresholds and
marks on both grids. The default fine-grid stream matches the 50,000-path
terminal-loss run. Outputs are written to
`numerics/output/terminal_mse_grid_check/`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/terminal_mse_grid_check.py
```
