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
