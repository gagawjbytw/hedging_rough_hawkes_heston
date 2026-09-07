# Variance-Optimal Hedging in the Rough Hawkes--Heston Model

This repository contains the numerical implementation accompanying the paper
[Variance-Optimal Hedging in the Rough Hawkes--Heston Model](https://github.com/gagawjbytw/hedging_rough_hawkes_heston).

## Requirements

Python 3.10 or later with the packages listed in [requirements.txt](requirements.txt).

## Usage

Run the original-market stability experiment from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/original_market_stability.py
```

Run the checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 numerics/test_original_market_stability.py
```

See [numerics/README.md](numerics/README.md) for command-line options and
output files.
