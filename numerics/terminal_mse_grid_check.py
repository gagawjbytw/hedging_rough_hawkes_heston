#!/usr/bin/env python3
"""Paired outer-grid check for terminal hedging MSE on original-market paths.

Fine Brownian normals are aggregated to the coarse grid. Both grids reuse the
same exponential Poisson thresholds and marks, so grid differences have paired
Monte Carlo standard errors. The default fine-grid random stream matches the
one used by hedging_performance_experiment.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

import hedging_performance_experiment as hedging
import original_market_stability as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "numerics" / "output" / "terminal_mse_grid_check"
REGULARIZED = "Kernel hedge eps/T=0.015625, R=80"
REFERENCE = "Unshifted-kernel hedge, R=80"
BLACK_SCHOLES = "Black--Scholes delta (implied-vol calibrated)"
STRATEGIES = ("No hedge", BLACK_SCHOLES, REGULARIZED, REFERENCE)


def grid_parameters(model: base.Model, maturity: float, steps: int):
    panels = np.array(
        [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0,
         30.0, 40.0, 60.0, 80.0]
    )
    lambdas, weights = base.fourier_rule(4, panels)
    u = 0.5 + 1j * lambdas
    epsilon = maturity / 64.0
    riccati = {
        REGULARIZED: base.solve_riccati_product_integration(
            model, u, maturity, steps, epsilon
        ),
        REFERENCE: base.solve_riccati_product_integration(
            model, u, maturity, steps, 0.0
        ),
    }
    capitals = {}
    for name, (_, forcing) in riccati.items():
        capitals[name] = float(
            base.capital_by_cutoff(
                base.initial_memory(model, forcing, maturity, steps),
                base.call_coefficients(lambdas, weights), lambdas,
                np.array([80.0]),
            )[0]
        )
    capitals["No hedge"] = capitals[REFERENCE]
    capitals[BLACK_SCHOLES] = capitals[REFERENCE]
    implied_vol = brentq(
        lambda vol: hedging.black_scholes_call(1.0, 1.0, maturity, vol)
        - capitals[REFERENCE],
        1.0e-5, 2.0,
    )
    return lambdas, weights, riccati, capitals, implied_vol


def evaluate_grid(
    model: base.Model,
    maturity: float,
    steps: int,
    sources: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    parameters,
):
    z1_fine, z2_fine, thresholds, marks = sources
    z1 = base.aggregate_normals(z1_fine, steps)
    z2 = base.aggregate_normals(z2_fine, steps)
    market = base.simulate_original_market(
        model, maturity, steps, z1, z2, thresholds, marks
    )
    lambdas, weights, riccati, _, implied_vol = parameters
    gains = {"No hedge": np.zeros(len(z1))}
    for name, epsilon in ((REGULARIZED, maturity / 64.0), (REFERENCE, 0.0)):
        _, gains[name] = hedging.evaluate_policy(
            model, maturity, market, lambdas, weights, 80.0,
            epsilon, riccati[name],
        )
    stock = np.asarray(market["stock"])
    dstock = np.asarray(market["dstock"])
    bs_gain = np.zeros(len(z1))
    for k in range(steps):
        tau = maturity * (steps - k) / steps
        stdev = implied_vol * math.sqrt(tau)
        d1 = np.log(stock[:, k]) / stdev + 0.5 * stdev
        bs_gain += ndtr(d1) * dstock[:, k]
    gains[BLACK_SCHOLES] = bs_gain
    payoff = np.maximum(stock[:, -1] - 1.0, 0.0)
    return payoff, gains, stock[:, -1], int(market["jump_count"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--fine-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.paths < 2 or args.batch_size < 1:
        raise ValueError("need at least two paths and a positive batch size")
    if args.fine_steps < 2 or args.fine_steps % 2:
        raise ValueError("--fine-steps must be even and at least two")

    model = base.Model()
    maturity = 0.25
    steps_list = (args.fine_steps // 2, args.fine_steps)
    parameters = {
        steps: grid_parameters(model, maturity, steps) for steps in steps_list
    }
    payoffs = {steps: [] for steps in steps_list}
    stocks = {steps: [] for steps in steps_list}
    gains = {
        steps: {name: [] for name in STRATEGIES} for steps in steps_list
    }
    jumps = {steps: 0 for steps in steps_list}
    child_seeds = np.random.SeedSequence(args.seed).spawn(
        math.ceil(args.paths / args.batch_size)
    )
    for batch_index, start in enumerate(range(0, args.paths, args.batch_size)):
        count = min(args.batch_size, args.paths - start)
        rng = np.random.default_rng(child_seeds[batch_index])
        sources = (
            rng.standard_normal((count, args.fine_steps), dtype=np.float64),
            rng.standard_normal((count, args.fine_steps), dtype=np.float64),
            np.cumsum(rng.exponential(size=(count, 8)), axis=1),
            rng.exponential(size=(count, 8)),
        )
        for steps in steps_list:
            payoff, batch_gains, terminal_stock, jump_count = evaluate_grid(
                model, maturity, steps, sources, parameters[steps]
            )
            payoffs[steps].append(payoff)
            stocks[steps].append(terminal_stock)
            jumps[steps] += jump_count
            for name in STRATEGIES:
                gains[steps][name].append(batch_gains[name])
        print(f"paths {start + count}/{args.paths}", flush=True)

    rows = []
    raw = {}
    for steps in steps_list:
        payoff = np.concatenate(payoffs[steps])
        grid_gains = {
            name: np.concatenate(gains[steps][name]) for name in STRATEGIES
        }
        capitals = parameters[steps][3]
        raw[steps] = (payoff, grid_gains, capitals)
        summary_rows, _ = hedging.summarize_errors(
            payoff, grid_gains, capitals, capitals[REFERENCE]
        )
        for row in summary_rows:
            rows.append({"steps": steps, **row})

    coarse, fine = steps_list
    paired = []
    for name in STRATEGIES:
        payoff_coarse, gains_coarse, capitals_coarse = raw[coarse]
        payoff_fine, gains_fine, capitals_fine = raw[fine]
        residual_coarse = payoff_coarse - gains_coarse[name]
        residual_fine = payoff_fine - gains_fine[name]
        method_difference = (
            (residual_fine - capitals_fine[name]) ** 2
            - (residual_coarse - capitals_coarse[name]) ** 2
        )
        centered_difference = (
            (residual_fine - np.mean(residual_fine)) ** 2
            - (residual_coarse - np.mean(residual_coarse)) ** 2
        )
        paired.append({
            "strategy": name,
            "fine_minus_coarse_method_mse": float(np.mean(method_difference)),
            "fine_minus_coarse_method_mse_se": base.standard_error(method_difference),
            "fine_minus_coarse_centered_mse": float(np.mean(centered_difference)),
            "fine_minus_coarse_centered_mse_se": base.standard_error(centered_difference),
        })

    diagnostic = {}
    for steps in steps_list:
        payoff, _, capitals = raw[steps]
        terminal_stock = np.concatenate(stocks[steps])
        diagnostic[str(steps)] = {
            "stock_mean": float(np.mean(terminal_stock)),
            "stock_mean_se": base.standard_error(terminal_stock),
            "payoff_mean": float(np.mean(payoff)),
            "payoff_mean_se": base.standard_error(payoff),
            "reference_fourier_capital": capitals[REFERENCE],
            "black_scholes_implied_vol": parameters[steps][4],
            "jump_count": jumps[steps],
        }
    payoff_difference = raw[fine][0] - raw[coarse][0]
    diagnostic["fine_minus_coarse_payoff_mean"] = float(np.mean(payoff_difference))
    diagnostic["fine_minus_coarse_payoff_mean_se"] = base.standard_error(
        payoff_difference
    )
    summary = {
        "configuration": {
            "paths": args.paths,
            "batch_size": args.batch_size,
            "coarse_steps": coarse,
            "fine_steps": fine,
            "maturity": maturity,
            "seed": args.seed,
            "strike": 1.0,
            "shared_sources": "fine Brownian normals aggregated; identical Poisson thresholds and marks",
        },
        "diagnostic": diagnostic,
        "results": rows,
        "paired_grid_differences": paired,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (OUT / "terminal_mse_by_grid.csv").open("w", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
