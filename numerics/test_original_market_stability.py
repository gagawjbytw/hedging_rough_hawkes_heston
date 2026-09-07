#!/usr/bin/env python3
"""Deterministic and fixed-history checks for original_market_stability.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad


MODULE_PATH = Path(__file__).with_name("original_market_stability.py")
SPEC = importlib.util.spec_from_file_location("original_market_stability", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main():
    model = MODULE.Model()
    maturity = 0.25
    epsilon = maturity / 32.0

    numerical_l1 = quad(
        lambda t: (
            t ** (model.alpha - 1.0)
            - (t + epsilon) ** (model.alpha - 1.0)
        )
        / MODULE.gamma(model.alpha),
        0.0,
        maturity,
        epsabs=1.0e-12,
    )[0]
    analytic_l1 = MODULE.kernel_l1_error(model.alpha, maturity, epsilon)
    assert abs(numerical_l1 - analytic_l1) < 1.0e-10

    psi_one, forcing_one = MODULE.solve_riccati_product_integration(
        model, np.array([1.0 + 0.0j]), maturity, 128, 0.0
    )
    assert np.max(np.abs(psi_one)) < 1.0e-13
    assert np.max(np.abs(forcing_one)) < 1.0e-13

    cutoffs = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
    panels = np.array(
        [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0,
         30.0, 40.0, 60.0, 80.0]
    )
    capitals = []
    for order in (4, 6):
        lambdas, weights = MODULE.fourier_rule(order, panels)
        u = 0.5 + 1j * lambdas
        _, forcing = MODULE.solve_riccati_product_integration(
            model, u, maturity, 256, 0.0
        )
        u0 = MODULE.initial_memory(model, forcing, maturity, 256)
        capitals.append(
            MODULE.capital_by_cutoff(
                u0, MODULE.call_coefficients(lambdas, weights), lambdas, cutoffs
            )
        )
    assert np.max(np.abs(capitals[0] - capitals[1])) < 4.0e-6

    z1, z2, thresholds, marks = MODULE.make_common_sources(16, 64, 12345, 8)
    market = MODULE.simulate_original_market(
        model, maturity, 64, z1, z2, thresholds, marks
    )
    reconstructed = MODULE.integrated_history_statistic(
        model, np.asarray(market["driver"]), maturity, 64, 0.0
    )
    assert np.max(
        np.abs(reconstructed - np.asarray(market["integrated_variance_raw"]))
    ) < 1.0e-12
    assert np.min(np.asarray(market["dintegrated_variance"])) >= 0.0

    print("original-market stability checks passed")


if __name__ == "__main__":
    main()
