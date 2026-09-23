# Decision-Consistent Variance-Optimal Hedging under Rough Volatility and Self-Exciting Jumps

This repository contains the numerical implementation and simulation outputs
accompanying the manuscript. The experiments cover original-market stability,
terminal hedging loss, and a paired time-grid check of terminal MSE.

## Requirements

Python 3.10 or later with the packages listed in [requirements.txt](requirements.txt).

## Usage

Run the three experiments from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/original_market_stability.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/hedging_performance_experiment.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/terminal_mse_grid_check.py
```

Run the checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/test_original_market_stability.py
```

See [numerics/README.md](numerics/README.md) for parameters, output files,
and the paired-grid construction. The reported CSV and JSON outputs are in
`numerics/output/`.
