#!/usr/bin/env python3
"""Correctness gate for weighted variational-GP classification objectives.

The NumPy oracle is independent of FortML's seeded RNG.  It evaluates a small
inducing-point logistic ELBO with a fixed normal table, verifies uniform weight
scaling and a nonuniform packed-gradient finite difference, then runs the
weighted FortML behavioral fixture.  Timings are correctness-gate durations,
not throughput measurements.
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
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def weighted_oracle() -> tuple[float, float, float, int]:
    """Return weighted value, scaling error, gradient FD error, and dimension."""
    x = np.array([-0.9, -0.2, 0.35, 1.0], dtype=np.float64)[:, None]
    inducing = np.array([-0.7, 0.55], dtype=np.float64)[:, None]
    labels = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64)
    weights = np.array([0.5, 1.5, 0.0, 2.0], dtype=np.float64)
    normal = np.array(
        [[-0.73, 0.41], [1.12, -0.28], [0.19, -1.31], [0.66, 0.07]],
        dtype=np.float64,
    )
    variance_scale, lengthscale, jitter = 1.4, 0.8, 1.0e-8

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = (left[:, None, :] - right[None, :, :]) ** 2
        return variance_scale * np.exp(-0.5 * distance.sum(axis=2) / lengthscale**2)

    prior = kernel(inducing, inducing)
    prior[np.diag_indices_from(prior)] += jitter
    prior_inverse = np.linalg.inv(prior)
    cross = kernel(inducing, x)
    projection = prior_inverse @ cross
    prior_marginal = np.diag(kernel(x, x)) - np.sum(projection * cross, axis=0)
    packed = np.array([0.20, -0.15, np.log(0.91), 0.10, np.log(1.07)])

    def evaluate(parameters: np.ndarray, gradient: bool = False):
        mean_u = parameters[:2]
        factor = np.array(
            [[np.exp(parameters[2]), 0.0],
             [parameters[3], np.exp(parameters[4])]], dtype=np.float64,
        )
        latent_mean = projection.T @ mean_u
        factor_projection = factor.T @ projection
        latent_variance = prior_marginal + np.sum(factor_projection**2, axis=0)
        latent = latent_mean[:, None] + np.sqrt(latent_variance)[:, None] * normal
        margins = labels[:, None] * latent
        log_likelihood = -np.logaddexp(0.0, -margins)
        likelihood = float(np.sum(weights[:, None] * log_likelihood) / normal.shape[1])
        solve_mean = prior_inverse @ mean_u
        solve_factor = prior_inverse @ factor
        kl = 0.5 * (
            np.sum(solve_factor * factor) + mean_u @ solve_mean - 2.0
            + np.linalg.slogdet(prior)[1] - 2.0 * (parameters[2] + parameters[4])
        )
        value = likelihood - kl
        if not gradient:
            return value, likelihood
        likelihood_gradient = labels[:, None] / (1.0 + np.exp(margins))
        mean_gradient = projection @ np.sum(weights[:, None] * likelihood_gradient,
                                             axis=1) / normal.shape[1]
        factor_gradient = np.zeros((2, 2), dtype=np.float64)
        for sample in range(x.shape[0]):
            z = factor_projection[:, sample]
            for draw in range(normal.shape[1]):
                derivative_scale = (
                    weights[sample] * likelihood_gradient[sample, draw]
                    * normal[sample, draw] / np.sqrt(latent_variance[sample])
                )
                factor_gradient += np.outer(projection[:, sample], z) * derivative_scale
        factor_gradient /= normal.shape[1]
        mean_gradient -= solve_mean
        factor_gradient -= solve_factor - np.linalg.inv(factor).T
        packed_gradient = np.array([
            mean_gradient[0], mean_gradient[1],
            factor_gradient[0, 0] * factor[0, 0], factor_gradient[1, 0],
            factor_gradient[1, 1] * factor[1, 1],
        ])
        return value, likelihood, packed_gradient

    value, likelihood, analytic = evaluate(packed, gradient=True)
    doubled_value, doubled_likelihood = evaluate(packed, gradient=False)
    # The doubled case is evaluated independently by scaling the row weights.
    old_weights = weights.copy()
    weights[:] = 2.0
    uniform_value, uniform_likelihood = evaluate(packed, gradient=False)
    weights[:] = old_weights
    scaling_error = max(
        abs(uniform_likelihood - 2.0 * float(np.sum(log_likelihood_for(packed)))),
        abs(uniform_value - (2.0 * float(np.sum(log_likelihood_for(packed))) -
                             (doubled_likelihood - doubled_value))),
    )
    step = 2.0e-6
    finite_difference = np.empty_like(packed)
    for index in range(packed.size):
        plus, _ = evaluate(packed + np.eye(1, packed.size, index)[0] * step)
        minus, _ = evaluate(packed - np.eye(1, packed.size, index)[0] * step)
        finite_difference[index] = (plus - minus) / (2.0 * step)
    gradient_error = float(np.max(np.abs(analytic - finite_difference)))
    if scaling_error > 2.0e-12 or gradient_error > 3.0e-7:
        raise RuntimeError(
            f"weighted variational-GP oracle failed: scale={scaling_error:.3e}, "
            f"gradient={gradient_error:.3e}"
        )
    return value, scaling_error, gradient_error, packed.size


def log_likelihood_for(parameters: np.ndarray) -> np.ndarray:
    """Independent unweighted likelihood table used by scaling assertion."""
    x = np.array([-0.9, -0.2, 0.35, 1.0])
    inducing = np.array([-0.7, 0.55])
    labels = np.array([-1.0, -1.0, 1.0, 1.0])
    normal = np.array([[-0.73, 0.41], [1.12, -0.28], [0.19, -1.31], [0.66, 0.07]])
    kernel = lambda left, right: 1.4 * np.exp(
        -0.5 * (left[:, None] - right[None, :]) ** 2 / 0.8**2)
    prior = kernel(inducing, inducing)
    prior[np.diag_indices_from(prior)] += 1.0e-8
    cross = kernel(inducing, x)
    projection = np.linalg.solve(prior, cross)
    marginal = np.diag(kernel(x, x)) - np.sum(projection * cross, axis=0)
    factor = np.array([[np.exp(parameters[2]), 0.0],
                       [parameters[3], np.exp(parameters[4])]])
    mean = projection.T @ parameters[:2]
    variance = marginal + np.sum((factor.T @ projection) ** 2, axis=0)
    margin = labels[:, None] * (mean[:, None] + np.sqrt(variance)[:, None] * normal)
    return -np.logaddexp(0.0, -margin).mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_variational_classification_weights.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    value, scaling_error, gradient_error, n_parameters = weighted_oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_variational_classification_weights",
                    "backend": "fortml", "device": "cpu", "n_samples": 4,
                    "n_features": 1, "n_inducing": 2,
                    "n_parameters": n_parameters})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", status="pass", seconds_per_operation="",
        metric="weighted_elbo", value=value,
        max_abs_error=max(scaling_error, gradient_error),
        oracle="independent NumPy weighted RBF ELBO, scaling, and packed-gradient FD",
        notes=f"uniform_scale_error={scaling_error:.3e}; gradient_error={gradient_error:.3e}")
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_gp_variational_classification_weights"],
                       cwd=fortml, check=True)
        status, notes = "pass", "weighted binary/OVR value-gradient-JVP and refusal gate"
    elapsed = time.perf_counter() - started
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="max_abs_error", value=max(scaling_error, gradient_error),
        max_abs_error=max(scaling_error, gradient_error),
        oracle="FortML test_gp_variational_classification_weights", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_weighted_ovr_graph", value="nan", max_abs_error="",
        oracle="typed FortML CUDA refusal",
        notes="weighted inducing solve and likelihood reduction are not resident")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
