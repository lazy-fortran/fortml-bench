#!/usr/bin/env python3
"""Correctness-only gate for inducing-point Bernoulli variational GP.

The NumPy fixture is deliberately independent of FortML's random-number
generator: it evaluates a small RBF inducing-point ELBO with a fixed standard
normal table and checks its analytic packed-parameter gradient by central
finite differences.  The Fortran release tests then check the same public
contract with their own seeded Monte Carlo table, parameter/query JVPs and
VJPs, minibatch, and CUDA refusal oracles.  The recorded wall time is a
correctness-gate duration, not a model or accelerator throughput measurement.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_inducing", "n_parameters", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return value + ("+dirty" if dirty else "")


def stable_logistic_margin(margin: np.ndarray) -> np.ndarray:
    """Return log(sigmoid(margin)) without overflow in either tail."""
    return -np.logaddexp(0.0, -margin)


def numpy_elbo_gradient() -> tuple[float, float, int]:
    """Evaluate an independent two-inducing-point ELBO and its FD gradient."""
    x = np.array([-0.9, -0.2, 0.35, 1.0], dtype=np.float64)[:, None]
    inducing = np.array([-0.7, 0.55], dtype=np.float64)[:, None]
    labels = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64)
    noise = np.array(
        [[-0.73, 0.41], [1.12, -0.28], [0.19, -1.31], [0.66, 0.07]],
        dtype=np.float64,
    )
    variance_scale = 1.4
    lengthscale = 0.8
    jitter = 1.0e-8

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = (left[:, None, :] - right[None, :, :]) ** 2
        return variance_scale * np.exp(-0.5 * distance.sum(axis=2) / lengthscale**2)

    prior = kernel(inducing, inducing)
    prior[np.diag_indices_from(prior)] += jitter
    prior_inverse = np.linalg.inv(prior)
    cross = kernel(inducing, x)
    projection = prior_inverse @ cross
    prior_marginal = np.diag(kernel(x, x)) - np.sum(projection * cross, axis=0)

    # [m(2), log(L11), L21, log(L22)] is the public packed convention.
    packed = np.array([0.20, -0.15, np.log(0.91), 0.10, np.log(1.07)], dtype=np.float64)

    def evaluate(parameters: np.ndarray, with_gradient: bool = False):
        mean_u = parameters[:2]
        factor = np.array(
            [[np.exp(parameters[2]), 0.0],
             [parameters[3], np.exp(parameters[4])]],
            dtype=np.float64,
        )
        latent_mean = projection.T @ mean_u
        factor_projection = factor.T @ projection
        latent_variance = prior_marginal + np.sum(factor_projection**2, axis=0)
        latent = latent_mean[:, None] + np.sqrt(latent_variance)[:, None] * noise
        margins = labels[:, None] * latent
        expected_log_likelihood = float(stable_logistic_margin(margins).mean())
        solve_mean = prior_inverse @ mean_u
        solve_factor = prior_inverse @ factor
        kl = 0.5 * (
            np.sum(solve_factor * factor) + mean_u @ solve_mean - 2.0
            + np.linalg.slogdet(prior)[1]
            - 2.0 * (parameters[2] + parameters[4])
        )
        value = expected_log_likelihood - kl
        if not with_gradient:
            return value

        likelihood_gradient = 1.0 / (1.0 + np.exp(margins)) * labels[:, None]
        mean_gradient = projection @ likelihood_gradient.mean(axis=1) / x.shape[0]
        factor_gradient = np.zeros((2, 2), dtype=np.float64)
        for sample in range(x.shape[0]):
            z = factor_projection[:, sample]
            for draw in range(noise.shape[1]):
                derivative_scale = (
                    likelihood_gradient[sample, draw] * noise[sample, draw]
                    / np.sqrt(latent_variance[sample])
                )
                factor_gradient += np.outer(projection[:, sample], z) * derivative_scale
        factor_gradient /= noise.shape[1] * x.shape[0]
        mean_gradient -= solve_mean
        factor_gradient -= solve_factor - np.linalg.inv(factor).T
        gradient = np.array(
            [mean_gradient[0], mean_gradient[1],
             factor_gradient[0, 0] * factor[0, 0], factor_gradient[1, 0],
             factor_gradient[1, 1] * factor[1, 1]],
            dtype=np.float64,
        )
        return value, gradient

    value, analytic = evaluate(packed, with_gradient=True)
    step = 2.0e-6
    finite_difference = np.empty_like(packed)
    for index in range(packed.size):
        plus = packed.copy()
        minus = packed.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (evaluate(plus) - evaluate(minus)) / (2.0 * step)
    error = float(np.max(np.abs(analytic - finite_difference)))
    if error > 2.0e-7 or not np.isfinite(value):
        raise RuntimeError(f"independent variational ELBO oracle failed: {error:.3e}")
    return value, error, packed.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/gp_variational_classification.csv"),
    )
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    ignored = (output,)
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": "gfortran",
        "flags": "-O3",
    }
    elbo, oracle_error, n_parameters = numpy_elbo_gradient()
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({
            "backend": "fortml", "device": "cpu", "n_samples": 4,
            "n_features": 1, "n_inducing": 2, "n_parameters": n_parameters,
        })
        row.update(values)
        rows.append(row)

    started = time.perf_counter()
    if args.skip_fortml:
        status = "skipped"
        notes = "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_gp_variational_classification"],
                       cwd=fortml, check=True)
        subprocess.run(["fo", "test", "test_gp_variational_classification_input"],
                       cwd=fortml, check=True)
        status = "pass"
        notes = ("Fortran tests supply seeded MC, parameter/query JVP/VJP, "
                 "minibatch, and refusal oracles")
    elapsed = time.perf_counter() - started
    add(workload="gp_variational_classification", phase="independent_oracle_gate",
        status=status, seconds_per_operation=elapsed, metric="elbo",
        value=elbo, max_abs_error=oracle_error,
        oracle="independent NumPy RBF ELBO and packed-gradient finite difference",
        notes=notes + "; correctness wall time, not throughput")

    add(workload="gp_variational_classification", phase="device_contract",
        device="cuda", status="unavailable", seconds_per_operation="",
        metric="device_supported", value=0.0, max_abs_error="",
        oracle="typed FortML CUDA capability refusal",
        notes="resident inducing solve, likelihood table, and reduction are not linked; no host fallback")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
