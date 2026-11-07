#!/usr/bin/env python3
"""Original-market kernel-regularization experiment.

The stochastic market is generated once with the fractional kernel K_alpha.
Every regularized hedge is then evaluated on that same (S,V,Z) history.  In
particular, no variance process or stock associated with K_epsilon is
simulated.  The script estimates the continuous-market L2(S) distance by
bracket quadrature and the uniform gain distance at the fixed cutoff R=20.

The outer time discretization follows the integrated-variance Euler idea of
Richard--Tan--Yang.  Brownian increments and the unit-rate Poisson time-change
variables are shared between the fine and coarse grids.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.signal import fftconvolve
from scipy.special import gamma


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "numerics" / "output" / "original_market_stability"


@dataclass(frozen=True)
class Model:
    alpha: float = 0.527
    rho: float = -0.731
    b: float = -1.812
    c: float = 0.115
    jump_leverage: float = 0.276
    beta0: float = 0.049
    v0: float = 0.0079
    s0: float = 1.0
    jump_intensity: float = 1.0
    jump_mean: float = 1.0

    def g0(self, t: np.ndarray | float) -> np.ndarray | float:
        return self.v0 + self.beta0 * np.asarray(t) ** self.alpha / gamma(
            self.alpha + 1.0
        )

    def integrated_g0(self, t: np.ndarray | float) -> np.ndarray | float:
        t_array = np.asarray(t)
        return (
            self.v0 * t_array
            + self.beta0 * t_array ** (self.alpha + 1.0) / gamma(self.alpha + 2.0)
        )


def affine_r(model: Model, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Riccati map for nu(dz)=kappa exp(-z/m) dz/m."""
    lam = model.jump_leverage
    mean = model.jump_mean
    jump = model.jump_intensity * (
        1.0 / (1.0 + mean * (lam * u - v))
        + u * mean * lam / (1.0 + mean * lam) - 1.0 - mean * v
    )
    return (
        0.5 * (u * u - u)
        + (model.b + model.rho * math.sqrt(model.c) * u) * v
        + 0.5 * model.c * v * v
        + jump
    )


def kernel_cell_integrals(
    alpha: float, maturity: float, steps: int, epsilon: float
) -> np.ndarray:
    """Integrals of K_epsilon over consecutive time cells."""
    h = maturity / steps
    right = epsilon + h * np.arange(1, steps + 1)
    left = epsilon + h * np.arange(steps)
    return (right**alpha - left**alpha) / gamma(alpha + 1.0)


def kernel_l1_error(alpha: float, maturity: float, epsilon: float) -> float:
    return float(
        (
            epsilon**alpha
            + maturity**alpha
            - (maturity + epsilon) ** alpha
        )
        / gamma(alpha + 1.0)
    )


def solve_riccati_product_integration(
    model: Model,
    u: np.ndarray,
    maturity: float,
    steps: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit product integration for psi=K_epsilon*R(u,psi)."""
    cell = kernel_cell_integrals(model.alpha, maturity, steps, epsilon)
    psi = np.zeros((steps + 1, len(u)), dtype=np.complex128)
    forcing = np.zeros_like(psi)
    forcing[0] = affine_r(model, u, psi[0])
    for k in range(1, steps + 1):
        psi[k] = cell[:k] @ forcing[k - 1 :: -1]
        forcing[k] = affine_r(model, u, psi[k])
        if not np.all(np.isfinite(psi[k])):
            raise FloatingPointError(
                f"Riccati solve failed at step {k}, epsilon={epsilon:g}"
            )
    return psi, forcing


def fourier_rule(order: int, cutoffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre panels whose right endpoints are the cutoffs."""
    nodes, weights = leggauss(order)
    left = 0.0
    all_nodes: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for right in cutoffs:
        all_nodes.append(0.5 * (right - left) * nodes + 0.5 * (right + left))
        all_weights.append(0.5 * (right - left) * weights)
        left = float(right)
    return np.concatenate(all_nodes), np.concatenate(all_weights)


def call_coefficients(lambdas: np.ndarray, quad_weights: np.ndarray) -> np.ndarray:
    """Positive-half Lewis coefficients for S0=K=1."""
    return quad_weights / (math.pi * (lambdas**2 + 0.25))


def projection_beta(model: Model, u: np.ndarray, psi_remaining: np.ndarray):
    lam = model.jump_leverage
    mean = model.jump_mean
    a = psi_remaining - lam * u
    jump_cov = model.jump_intensity * (
        1.0 / (1.0 + mean * (lam - a))
        - 1.0 / (1.0 - mean * a)
        - 1.0 / (1.0 + mean * lam)
        + 1.0
    )
    d_stock = 1.0 + model.jump_intensity * (
        1.0 + 1.0 / (1.0 + 2.0 * mean * lam) - 2.0 / (1.0 + mean * lam)
    )
    beta = (u + model.rho * math.sqrt(model.c) * psi_remaining + jump_cov) / d_stock
    return beta, d_stock


def clipped_exponential(z: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(np.real(z), 0.0) + 1j * np.imag(z))


def initial_memory(
    model: Model, forcing: np.ndarray, maturity: float, steps: int
) -> np.ndarray:
    """Left-cell quadrature of integral_0^T F(T-s)g0(s)ds."""
    h = maturity / steps
    times = h * np.arange(steps)
    remaining = steps - np.arange(steps)
    return h * np.sum(forcing[remaining] * model.g0(times)[:, None], axis=0)


def capital_by_cutoff(
    u0: np.ndarray,
    coefficients: np.ndarray,
    lambdas: np.ndarray,
    cutoffs: np.ndarray,
) -> np.ndarray:
    values = np.empty(len(cutoffs))
    exp_u0 = clipped_exponential(u0)
    for j, cutoff in enumerate(cutoffs):
        mask = lambdas <= cutoff + 1.0e-12
        values[j] = 1.0 - np.real(np.sum(coefficients[mask] * exp_u0[mask]))
    return values


def make_common_sources(paths: int, fine_steps: int, seed: int, max_events: int):
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((paths, fine_steps), dtype=np.float64)
    z2 = rng.standard_normal((paths, fine_steps), dtype=np.float64)
    thresholds = np.cumsum(rng.exponential(size=(paths, max_events)), axis=1)
    marks = rng.exponential(size=(paths, max_events))
    return z1, z2, thresholds, marks


def aggregate_normals(source: np.ndarray, steps: int) -> np.ndarray:
    fine_steps = source.shape[1]
    if fine_steps % steps:
        raise ValueError("fine step count must be divisible by requested steps")
    ratio = fine_steps // steps
    if ratio == 1:
        return source
    return source.reshape(len(source), steps, ratio).sum(axis=2) / math.sqrt(ratio)


def simulate_original_market(
    model: Model,
    maturity: float,
    steps: int,
    z1: np.ndarray,
    z2: np.ndarray,
    thresholds: np.ndarray,
    marks: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    """Integrated-variance Euler simulation under the original kernel."""
    paths = len(z1)
    h = maturity / steps
    lags = h * np.arange(1, steps + 1)
    kernel_values = lags ** (model.alpha - 1.0) / gamma(model.alpha)
    driver = np.zeros((paths, steps + 1), dtype=np.float64)
    integrated_raw = np.zeros((paths, steps + 1), dtype=np.float64)
    integrated = np.zeros_like(integrated_raw)
    x = np.zeros((paths, steps + 1), dtype=np.float64)
    martingale_one = np.zeros(paths)
    martingale_two = np.zeros(paths)
    cumulative_marks = np.zeros(paths)
    jump_count = 0
    q_mean = model.jump_intensity * (
        1.0 / (1.0 + model.jump_mean * model.jump_leverage) - 1.0
    )
    sqrt_independent = math.sqrt(1.0 - model.rho**2)

    for k in range(steps):
        time_next = (k + 1) * h
        raw_next = model.integrated_g0(time_next) + h * (
            driver[:, : k + 1] @ kernel_values[k::-1]
        )
        integrated_raw[:, k + 1] = raw_next
        integrated[:, k + 1] = np.maximum(integrated[:, k], raw_next)
        delta_integrated = integrated[:, k + 1] - integrated[:, k]

        event_mask = (thresholds > model.jump_intensity * integrated[:, k, None]) & (
            thresholds <= model.jump_intensity * integrated[:, k + 1, None]
        )
        jump_sum = model.jump_mean * np.sum(event_mask * marks, axis=1)
        jump_count += int(np.count_nonzero(event_mask))

        sqrt_delta = np.sqrt(delta_integrated)
        martingale_one += sqrt_delta * z1[:, k]
        martingale_two += sqrt_delta * z2[:, k]
        cumulative_marks += jump_sum
        driver[:, k + 1] = (
            (model.b - model.jump_intensity * model.jump_mean) * integrated[:, k + 1]
            + math.sqrt(model.c) * martingale_two
            + cumulative_marks
        )
        x[:, k + 1] = (
            (-0.5 - q_mean) * integrated[:, k + 1]
            + sqrt_independent * martingale_one
            + model.rho * martingale_two
            - model.jump_leverage * cumulative_marks
        )

    stock = model.s0 * np.exp(x)
    if np.any(model.jump_intensity * integrated[:, -1] >= thresholds[:, -1]):
        raise RuntimeError("increase --max-events: a path exhausted Poisson thresholds")
    return {
        "x": x,
        "stock": stock,
        "dstock": np.diff(stock, axis=1),
        "driver": driver,
        "dz": np.diff(driver, axis=1),
        "integrated_variance_raw": integrated_raw,
        "integrated_variance": integrated,
        "dintegrated_variance": np.diff(integrated, axis=1),
        "jump_count": jump_count,
    }


def integrated_history_statistic(
    model: Model,
    driver: np.ndarray,
    maturity: float,
    steps: int,
    epsilon: float,
) -> np.ndarray:
    """Compute int_0^t Vhat_epsilon(s)ds from the fixed driver history."""
    h = maturity / steps
    lags = h * np.arange(1, steps + 1)
    kernel_values = (lags + epsilon) ** (model.alpha - 1.0) / gamma(model.alpha)
    causal_kernel = np.concatenate(([0.0], kernel_values))
    convolution = h * fftconvolve(
        driver, causal_kernel[None, :], mode="full", axes=1
    )[:, : steps + 1]
    return convolution + model.integrated_g0(
        h * np.arange(steps + 1)
    )[None, :]


def standard_error(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def evaluate_strategies(
    model: Model,
    maturity: float,
    steps: int,
    market: dict[str, np.ndarray | float | int],
    lambdas: np.ndarray,
    quad_weights: np.ndarray,
    cutoffs: np.ndarray,
    epsilons: np.ndarray,
    riccati: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray | float]:
    """Evaluate all kernels on the same original-market path ensemble."""
    x = np.asarray(market["x"])
    stock = np.asarray(market["stock"])
    dstock = np.asarray(market["dstock"])
    driver = np.asarray(market["driver"])
    dz = np.asarray(market["dz"])
    dintegrated_variance = np.asarray(market["dintegrated_variance"])
    paths = len(x)
    u = 0.5 + 1j * lambdas
    coefficients = call_coefficients(lambdas, quad_weights)
    n_kernels = 1 + len(epsilons)
    n_cutoffs = len(cutoffs)
    n_nodes = len(lambdas)

    integrated_histories = [
        integrated_history_statistic(model, driver, maturity, steps, epsilon)
        for epsilon in np.concatenate(([0.0], epsilons))
    ]
    integrated_history_increments = [
        np.diff(history, axis=1) for history in integrated_histories
    ]
    u0 = [initial_memory(model, forcing, maturity, steps) for _, forcing in riccati]
    u_state = [np.broadcast_to(value, (paths, n_nodes)).copy() for value in u0]

    l2_same = np.zeros((paths, len(epsilons), n_cutoffs))
    gain_same = np.zeros_like(l2_same)
    gain_same_sup = np.zeros_like(l2_same)

    masks = [lambdas <= cutoff + 1.0e-12 for cutoff in cutoffs]
    d_stock = projection_beta(model, u, riccati[0][0][steps])[1]

    for k in range(steps):
        remaining = steps - k
        theta: list[np.ndarray] = []
        for j in range(n_kernels):
            psi, _ = riccati[j]
            beta, _ = projection_beta(model, u, psi[remaining])
            transformed = clipped_exponential(u_state[j])
            h_over_s = np.exp((u[None, :] - 1.0) * x[:, k, None]) * transformed
            node_value = h_over_s * beta[None, :] * coefficients[None, :]
            theta_j = np.empty((paths, n_cutoffs))
            for r, mask in enumerate(masks):
                theta_j[:, r] = 1.0 - np.real(np.sum(node_value[:, mask], axis=1))
            theta.append(theta_j)

        # Quadrature weight for the continuous-market L2(S) norm. The exact
        # discrete stock bracket is stock[:, k]**2 * expm1(D * Delta Abar).
        bracket_weight = d_stock * stock[:, k] ** 2 * dintegrated_variance[:, k]
        for e in range(len(epsilons)):
            same_difference = theta[e + 1] - theta[0]
            l2_same[:, e, :] += bracket_weight[:, None] * same_difference**2
            gain_same[:, e, :] += same_difference * dstock[:, k, None]
            gain_same_sup[:, e, :] = np.maximum(
                gain_same_sup[:, e, :], gain_same[:, e, :] ** 2
            )

        for j in range(n_kernels):
            psi, forcing = riccati[j]
            u_state[j] += (
                psi[remaining][None, :] * dz[:, k, None]
                - forcing[remaining][None, :]
                * integrated_history_increments[j][:, k, None]
            )

    same_mean = np.mean(l2_same, axis=0)
    same_se = np.std(l2_same, axis=0, ddof=1) / math.sqrt(paths)
    gain_same_mean = np.mean(gain_same_sup, axis=0)
    gain_same_se = np.std(gain_same_sup, axis=0, ddof=1) / math.sqrt(paths)

    return {
        "l2_same_mean": same_mean,
        "l2_same_se": same_se,
        "gain_same_mean": gain_same_mean,
        "gain_same_se": gain_same_se,
        "stock_mean": float(np.mean(stock[:, -1])),
        "stock_se": standard_error(stock[:, -1]),
    }


def run_grid(
    model: Model,
    maturity: float,
    steps: int,
    z1_fine: np.ndarray,
    z2_fine: np.ndarray,
    thresholds: np.ndarray,
    marks: np.ndarray,
    lambdas: np.ndarray,
    quad_weights: np.ndarray,
    cutoffs: np.ndarray,
    epsilon_ratios: np.ndarray,
) -> dict:
    z1 = aggregate_normals(z1_fine, steps)
    z2 = aggregate_normals(z2_fine, steps)
    epsilons = maturity * epsilon_ratios
    u = 0.5 + 1j * lambdas
    riccati = [
        solve_riccati_product_integration(model, u, maturity, steps, epsilon)
        for epsilon in np.concatenate(([0.0], epsilons))
    ]
    market = simulate_original_market(
        model, maturity, steps, z1, z2, thresholds, marks
    )
    strategy = evaluate_strategies(
        model,
        maturity,
        steps,
        market,
        lambdas,
        quad_weights,
        cutoffs,
        epsilons,
        riccati,
    )
    return {
        "paths": len(z1),
        "steps": steps,
        "lambdas": lambdas,
        "cutoffs": cutoffs,
        "epsilons": epsilons,
        "jump_count": int(market["jump_count"]),
        **strategy,
    }


def write_outputs(
    args: argparse.Namespace,
    model: Model,
    cutoffs: np.ndarray,
    epsilon_ratios: np.ndarray,
    results: list[dict],
):
    """Write the fixed-cutoff convergence results used in Section 5."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        for e, ratio in enumerate(epsilon_ratios):
            rows.append({
                "steps": result["steps"],
                "epsilon_over_T": ratio,
                "epsilon": result["epsilons"][e],
                "cutoff": float(cutoffs[0]),
                "strategy_l2_same_cutoff": result["l2_same_mean"][e, 0],
                "strategy_l2_same_cutoff_se": result["l2_same_se"][e, 0],
                "gain_sup_same_cutoff": result["gain_same_mean"][e, 0],
                "gain_sup_same_cutoff_se": result["gain_same_se"][e, 0],
            })
    with (OUTPUT / "convergence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "configuration": {
            "paths": args.paths,
            "batch_size": args.batch_size,
            "fine_steps": args.steps,
            "coarse_steps": args.steps // 2,
            "maturity": args.maturity,
            "seed": args.seed,
            "fourier_order_per_panel": args.fourier_order,
            "fixed_cutoff": float(cutoffs[0]),
            "epsilon_over_T": epsilon_ratios.tolist(),
        },
        "model": asdict(model),
        "stock_diagnostics": [
            {"steps": r["steps"], "stock_mean": r["stock_mean"],
             "stock_se": r["stock_se"], "jump_count": r["jump_count"]}
            for r in results
        ],
    }
    with (OUTPUT / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--maturity", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--fourier-order", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 128 or args.steps & (args.steps - 1):
        raise ValueError("--steps must be a power of two and at least 128")
    if args.paths < 2:
        raise ValueError("at least two paths are required")
    np.seterr(over="raise", invalid="raise", divide="raise")
    model = Model()
    cutoffs = np.array([20.0])
    fourier_panels = np.array(
        [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0]
    )
    epsilon_ratios = np.array([0.25, 0.125, 0.0625, 0.03125, 0.015625])
    lambdas, quad_weights = fourier_rule(args.fourier_order, fourier_panels)
    z1, z2, thresholds, marks = make_common_sources(
        args.paths, args.steps, args.seed, args.max_events
    )
    results = []
    for steps in (args.steps // 2, args.steps):
        result_batches = []
        for start in range(0, args.paths, args.batch_size):
            stop = min(start + args.batch_size, args.paths)
            print(
                f"grid={steps}: paths {start + 1}-{stop} of {args.paths}",
                flush=True,
            )
            result_batches.append(
                run_grid(
                    model,
                    args.maturity,
                    steps,
                    z1[start:stop],
                    z2[start:stop],
                    thresholds[start:stop],
                    marks[start:stop],
                    lambdas,
                    quad_weights,
                    cutoffs,
                    epsilon_ratios,
                )
            )
        # Combine batch estimates exactly by rerunning the strategy aggregation
        # is unnecessary: store pathwise-independent weighted first moments and
        # pool standard errors through the raw second-moment reconstruction.
        results.append(pool_batches(result_batches, stop_count=args.paths))
    write_outputs(args, model, cutoffs, epsilon_ratios, results)
    print(json.dumps(json.loads((OUTPUT / "summary.json").read_text()), indent=2))


def pool_mean_se(means: list[np.ndarray], ses: list[np.ndarray], sizes: list[int]):
    total = sum(sizes)
    sum1 = sum(size * mean for size, mean in zip(sizes, means))
    # Recover within-batch sum of squares, then add between-batch variation.
    sum2 = sum(
        (size - 1) * (se * math.sqrt(size)) ** 2 + size * mean**2
        for size, mean, se in zip(sizes, means, ses)
    )
    mean = sum1 / total
    variance = np.maximum((sum2 - total * mean**2) / (total - 1), 0.0)
    return mean, np.sqrt(variance / total)


def pool_batches(batches: list[dict], stop_count: int) -> dict:
    sizes = [int(batch["paths"]) for batch in batches]
    if sum(sizes) != stop_count:
        raise ValueError("pooled path count does not match requested path count")
    pooled = dict(batches[0])
    for mean_key, se_key in (
        ("l2_same_mean", "l2_same_se"),
        ("gain_same_mean", "gain_same_se"),
    ):
        pooled[mean_key], pooled[se_key] = pool_mean_se(
            [np.asarray(batch[mean_key]) for batch in batches],
            [np.asarray(batch[se_key]) for batch in batches],
            sizes,
        )
    pooled["jump_count"] = sum(int(batch["jump_count"]) for batch in batches)
    stock_mean, stock_se = pool_mean_se(
        [np.asarray(batch["stock_mean"]) for batch in batches],
        [np.asarray(batch["stock_se"]) for batch in batches],
        sizes,
    )
    pooled["stock_mean"] = float(stock_mean)
    pooled["stock_se"] = float(stock_se)
    return pooled


if __name__ == "__main__":
    main()
