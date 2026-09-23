#!/usr/bin/env python3
"""Terminal-loss comparison on common paths for the manuscript experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

import original_market_stability as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "numerics" / "output" / "hedging_performance_experiment"


def black_scholes_call(s0: float, strike: float, tau: float, vol: float) -> float:
    if tau <= 0.0:
        return max(s0 - strike, 0.0)
    stdev = vol * math.sqrt(tau)
    d1 = (math.log(s0 / strike) + 0.5 * stdev**2) / stdev
    d2 = d1 - stdev
    return s0 * ndtr(d1) - strike * ndtr(d2)


def evaluate_policy(
    model: base.Model,
    maturity: float,
    market: dict,
    lambdas: np.ndarray,
    weights: np.ndarray,
    cutoff: float,
    epsilon: float,
    riccati: tuple[np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray]:
    """Return Fourier capital and pathwise terminal gain for one hedge."""
    x = np.asarray(market["x"])
    stock = np.asarray(market["stock"])
    dstock = np.asarray(market["dstock"])
    driver = np.asarray(market["driver"])
    dz = np.asarray(market["dz"])
    history = base.integrated_history_statistic(
        model, driver, maturity, stock.shape[1] - 1, epsilon
    )
    dhistory = np.diff(history, axis=1)
    psi, forcing = riccati
    u = 0.5 + 1j * lambdas
    coefficients = base.call_coefficients(lambdas, weights)
    mask = lambdas <= cutoff + 1.0e-12
    capital = base.capital_by_cutoff(
        base.initial_memory(model, forcing, maturity, stock.shape[1] - 1),
        coefficients,
        lambdas,
        np.array([cutoff]),
    )[0]
    u_state = np.broadcast_to(
        base.initial_memory(model, forcing, maturity, stock.shape[1] - 1),
        (len(stock), len(lambdas)),
    ).copy()
    gain = np.zeros(len(stock))

    for k in range(stock.shape[1] - 1):
        remaining = stock.shape[1] - 1 - k
        beta, _ = base.projection_beta(model, u, psi[remaining])
        transformed = base.clipped_exponential(u_state)
        h_over_s = np.exp((u[None, :] - 1.0) * x[:, k, None]) * transformed
        node_value = h_over_s * beta[None, :] * coefficients[None, :]
        theta = 1.0 - np.real(np.sum(node_value[:, mask], axis=1))
        gain += theta * dstock[:, k]
        u_state += (
            psi[remaining][None, :] * dz[:, k, None]
            - forcing[remaining][None, :] * dhistory[:, k, None]
        )
    return float(capital), gain


def summarize_errors(
    payoff: np.ndarray,
    gains: dict[str, np.ndarray],
    capitals: dict[str, float],
    reference_capital: float,
) -> tuple[list[dict], list[dict]]:
    baseline_centered = payoff - np.mean(payoff)
    baseline_mse = float(np.mean(baseline_centered**2))
    names = list(gains)
    rows = []
    centered_squares = {}
    for name in names:
        raw = payoff - gains[name]
        optimal_sample_capital = float(np.mean(raw))
        centered_error = raw - optimal_sample_capital
        centered_square = centered_error**2
        centered_squares[name] = centered_square
        centered_mse = float(np.mean(centered_square))
        centered_se = float(np.std(centered_square, ddof=1) / math.sqrt(len(raw)))

        method_capital = float(capitals[name])
        method_error = payoff - method_capital - gains[name]
        method_square = method_error**2
        method_mse = float(np.mean(method_square))
        method_se = float(np.std(method_square, ddof=1) / math.sqrt(len(raw)))

        paired_improvement = baseline_centered**2 - centered_square
        paired_se = float(
            np.std(paired_improvement, ddof=1) / math.sqrt(len(raw))
        )
        rows.append(
            {
                "strategy": name,
                "capital_from_method": method_capital,
                "best_sample_capital_for_strategy": optimal_sample_capital,
                "centered_terminal_mse": centered_mse,
                "centered_terminal_mse_se": centered_se,
                "variance_reduction_vs_no_hedge": 1.0 - centered_mse / baseline_mse,
                "paired_mse_improvement_vs_no_hedge": float(
                    np.mean(paired_improvement)
                ),
                "paired_mse_improvement_se": paired_se,
                "method_capital_terminal_mse": method_mse,
                "method_capital_terminal_mse_se": method_se,
                "method_capital_mse_vs_reference_capital": float(
                    np.mean((payoff - reference_capital - gains[name]) ** 2)
                ),
            }
        )
    delta_name = "Black--Scholes delta (implied-vol calibrated)"
    comparisons = []
    for candidate in names:
        for benchmark in ("No hedge", delta_name, "Unshifted-kernel hedge, R=80"):
            if candidate == benchmark:
                continue
            improvement = centered_squares[benchmark] - centered_squares[candidate]
            comparisons.append(
                {
                    "candidate": candidate,
                    "benchmark": benchmark,
                    "mse_improvement_candidate_minus_benchmark": float(
                        np.mean(improvement)
                    ),
                    "paired_se": float(
                        np.std(improvement, ddof=1) / math.sqrt(len(improvement))
                    ),
                    "positive_means_candidate_is_better": True,
                }
            )
    return rows, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    paths, batch_size, steps, maturity, seed = (
        args.paths,
        args.batch_size,
        args.steps,
        0.25,
        args.seed,
    )
    model = base.Model()
    cutoffs = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
    epsilon_ratios = np.array([0.25, 0.125, 0.0625, 0.03125, 0.015625])
    panels = np.array(
        [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0,
         30.0, 40.0, 60.0, 80.0]
    )
    lambdas, weights = base.fourier_rule(4, panels)
    u = 0.5 + 1j * lambdas
    epsilons = maturity * epsilon_ratios
    riccati = [
        base.solve_riccati_product_integration(model, u, maturity, steps, epsilon)
        for epsilon in np.concatenate(([0.0], epsilons))
    ]

    capitals: dict[str, float] = {}
    gain_parts: dict[str, list[np.ndarray]] = {"No hedge": []}
    for e, r in enumerate(range(len(cutoffs))):
        name = f"Kernel hedge eps/T={epsilon_ratios[e]:g}, R={cutoffs[r]:g}"
        capitals[name] = float(
            base.capital_by_cutoff(
                base.initial_memory(model, riccati[e + 1][1], maturity, steps),
                base.call_coefficients(lambdas, weights),
                lambdas,
                np.array([cutoffs[r]]),
            )[0]
        )
        gain_parts[name] = []

    reference_name = "Unshifted-kernel hedge, R=80"
    capitals[reference_name] = float(
        base.capital_by_cutoff(
            base.initial_memory(model, riccati[0][1], maturity, steps),
            base.call_coefficients(lambdas, weights),
            lambdas,
            np.array([80.0]),
        )[0]
    )
    gain_parts[reference_name] = []
    bs_name = "Black--Scholes delta (implied-vol calibrated)"
    gain_parts[bs_name] = []

    reference_capital = capitals[reference_name]
    implied_vol = brentq(
        lambda vol: black_scholes_call(1.0, 1.0, maturity, vol)
        - reference_capital,
        1.0e-5,
        2.0,
    )
    payoff_parts = []
    stock_terminal_parts = []
    total_jumps = 0
    child_seeds = np.random.SeedSequence(seed).spawn(math.ceil(paths / batch_size))
    for batch_index, start in enumerate(range(0, paths, batch_size)):
        batch_paths = min(batch_size, paths - start)
        rng = np.random.default_rng(child_seeds[batch_index])
        z1 = rng.standard_normal((batch_paths, steps), dtype=np.float64)
        z2 = rng.standard_normal((batch_paths, steps), dtype=np.float64)
        thresholds = np.cumsum(rng.exponential(size=(batch_paths, 8)), axis=1)
        marks = rng.exponential(size=(batch_paths, 8))
        market = base.simulate_original_market(
            model, maturity, steps, z1, z2, thresholds, marks
        )
        total_jumps += int(market["jump_count"])
        for e, r in enumerate(range(len(cutoffs))):
            name = f"Kernel hedge eps/T={epsilon_ratios[e]:g}, R={cutoffs[r]:g}"
            _, batch_gain = evaluate_policy(
                model,
                maturity,
                market,
                lambdas,
                weights,
                float(cutoffs[r]),
                float(epsilons[e]),
                riccati[e + 1],
            )
            gain_parts[name].append(batch_gain)
        _, reference_gain = evaluate_policy(
            model, maturity, market, lambdas, weights, 80.0, 0.0, riccati[0]
        )
        gain_parts[reference_name].append(reference_gain)

        stock = np.asarray(market["stock"])
        dstock = np.asarray(market["dstock"])
        bs_gain = np.zeros(batch_paths)
        for k in range(steps):
            tau = maturity - k * maturity / steps
            stdev = implied_vol * math.sqrt(tau)
            d1 = np.log(stock[:, k]) / stdev + 0.5 * stdev
            bs_gain += ndtr(d1) * dstock[:, k]
        gain_parts[bs_name].append(bs_gain)
        payoff_parts.append(np.maximum(stock[:, -1] - 1.0, 0.0))
        stock_terminal_parts.append(stock[:, -1])

    payoff = np.concatenate(payoff_parts)
    stock_terminal = np.concatenate(stock_terminal_parts)
    gains = {
        name: np.concatenate(parts) if parts else np.zeros(paths)
        for name, parts in gain_parts.items()
    }
    capitals["No hedge"] = reference_capital
    capitals[bs_name] = reference_capital

    rows, comparisons = summarize_errors(payoff, gains, capitals, reference_capital)
    for row in rows:
        if row["strategy"].startswith("Kernel hedge"):
            idx = int(row["strategy"].split("R=")[-1])
            # The ordered diagonal uses the five cutoff values as indices.
            r = list(cutoffs).index(float(idx))
            row["epsilon_over_T"] = float(epsilon_ratios[r])
            row["cutoff"] = float(cutoffs[r])
    summary = {
        "configuration": {
            "paths": paths,
            "batch_size": batch_size,
            "steps": steps,
            "maturity": maturity,
            "seed": seed,
            "strike": 1.0,
            "stock_mean": float(np.mean(stock_terminal)),
            "stock_mean_se": base.standard_error(stock_terminal),
            "payoff_mean_mc": float(np.mean(payoff)),
            "payoff_mean_se": base.standard_error(payoff),
            "unshifted_R80_fourier_capital": reference_capital,
            "black_scholes_implied_vol": implied_vol,
            "mean_jumps": total_jumps / paths,
            "total_jumps": total_jumps,
        },
        "results": rows,
        "paired_centered_mse_comparisons": comparisons,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (OUT / "terminal_mse.csv").open("w", newline="") as handle:
        fieldnames = list(rows[0]) + ["epsilon_over_T", "cutoff"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
