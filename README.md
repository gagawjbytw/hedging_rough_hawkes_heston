# Variance-Optimal Hedging in the Rough Hawkes--Heston Model

This repository contains the numerical implementation and simulation outputs
accompanying the manuscript. The experiments cover original-market stability,
terminal hedging loss, a paired time-grid check of terminal MSE, and public
SPX calibration against a rough Heston control on four snapshots.

## Requirements

Python 3.11 or later with the packages listed in [requirements.txt](requirements.txt).

```bash
python3 -m pip install -r requirements.txt
```

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

Run the public SPX calibration and its numerical checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/calibrate_public_spx.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/test_public_spx_calibration.py
python3 numerics/summarize_public_spx.py
```

Generate the manuscript figures from the saved results:

```bash
mkdir -p arxiv/figures
python3 numerics/plot_hedging_convergence_paper.py
python3 numerics/plot_public_spx_paper.py
```

The raw public SPX snapshots and their pinned source URLs and SHA-256 checksums
are in `numerics/data/public_spx/`. Calibration outputs, fitted parameters,
held-out predictions, and numerical validation results are in
`numerics/output/public_spx_calibration/`.

See [numerics/README.md](numerics/README.md) for parameters, output files,
and the paired-grid construction. The reported CSV and JSON outputs are in
`numerics/output/`.
