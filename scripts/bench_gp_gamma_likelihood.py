#!/usr/bin/env python3
"""Benchmark weighted Gamma GP likelihood products and shape fitting."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.special import digamma, polygamma


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "scipy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    observations = np.array([0.35, 0.8, 1.7, 3.2, 5.5, 2.4])
    latents = np.log(np.array([0.5, 1.0, 1.4, 2.8, 4.1, 2.0]))
    weights = np.array([1.0, 0.25, 2.0, 1.5, 0.0, 0.75])
    direction = np.array([0.3, -0.2, 0.1, 0.45, -0.4, 0.25, -0.35])
    return observations, latents, weights, direction, math.log(1.7)


def oracle_value(observations: np.ndarray, latents: np.ndarray,
                 weights: np.ndarray, log_shape: float) -> float:
    shape = math.exp(log_shape)
    terms = (
        shape * log_shape - math.lgamma(shape)
        + (shape - 1.0) * np.log(observations)
        - shape * latents - shape * observations / np.exp(latents)
    )
    return float(np.dot(weights, terms))


def oracle_gradient(observations: np.ndarray, coordinates: np.ndarray,
                    weights: np.ndarray, step: float) -> np.ndarray:
    n_samples = observations.size
    gradient = np.empty_like(coordinates)
    for index in range(coordinates.size):
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (
            oracle_value(observations, plus[:n_samples], weights, plus[-1])
            - oracle_value(observations, minus[:n_samples], weights, minus[-1])
        ) / (2.0 * step)
    return gradient


def numpy_products(observations: np.ndarray, latents: np.ndarray,
                   weights: np.ndarray, log_shape: float,
                   direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = math.exp(log_shape)
    ratio = observations * np.exp(-latents)
    derivative_shape = (
        log_shape + 1.0 - digamma(shape) + np.log(observations)
        - latents - ratio
    )
    gradient = np.empty(latents.size + 1)
    gradient[:-1] = weights * shape * (ratio - 1.0)
    gradient[-1] = np.sum(weights * shape * derivative_shape)
    diagonal = -weights * shape * ratio
    mixed = weights * shape * (ratio - 1.0)
    shape_hessian = np.sum(weights * (
        shape * derivative_shape + shape - shape * shape * polygamma(1, shape)
    ))
    product = np.empty_like(gradient)
    product[:-1] = diagonal * direction[:-1] + mixed * direction[-1]
    product[-1] = np.dot(mixed, direction[:-1]) + shape_hessian * direction[-1]
    return gradient, product


def parse_probe(stdout: str) -> dict[str, float | int]:
    records: dict[str, float | int] = {}
    for line in stdout.splitlines():
        if not line.startswith("gamma_likelihood_"):
            continue
        key, raw = line.strip().split(",", maxsplit=1)
        if key.endswith(("repeats", "iterations", "status")):
            records[key] = int(raw)
        else:
            records[key] = float(raw)
    required = {
        "gamma_likelihood_repeats", "gamma_likelihood_product_seconds",
        "gamma_likelihood_value", "gamma_likelihood_shape_gradient",
        "gamma_likelihood_shape_hvp", "gamma_likelihood_fit_seconds",
        "gamma_likelihood_fitted_log_shape",
        "gamma_likelihood_fit_gradient_norm",
        "gamma_likelihood_fit_iterations", "gamma_likelihood_cuda_status",
    }
    required.update(f"gamma_likelihood_latent_gradient_{i}" for i in range(1, 7))
    required.update(f"gamma_likelihood_latent_hvp_{i}" for i in range(1, 7))
    missing = required - records.keys()
    if missing:
        raise RuntimeError(f"Gamma likelihood probe omitted {sorted(missing)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_gamma_likelihood.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/GP_GAMMA_LIKELIHOOD.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    environment = os.environ.copy()
    environment["FO_SCAN_FALLBACK"] = "regex"
    environment["FO_FC"] = "gfortran"
    subprocess.run(["fo", "test", "test_gamma_likelihood"], cwd=fortml,
                   env=environment, check=True)
    probe = subprocess.run(
        ["fo", "exec", "fortml_bench_gamma_likelihood"], cwd=fortml,
        env=environment, check=True, text=True, capture_output=True,
    )
    record = parse_probe(probe.stdout)
    observations, latents, weights, direction, log_shape = fixture()
    coordinates = np.concatenate((latents, [log_shape]))
    gradient_step = 2.0e-6
    direction_step = 2.0e-4
    gradient_oracle = oracle_gradient(
        observations, coordinates, weights, gradient_step,
    )
    plus = coordinates + direction_step * direction
    minus = coordinates - direction_step * direction
    hvp_oracle = (
        oracle_gradient(observations, plus, weights, gradient_step)
        - oracle_gradient(observations, minus, weights, gradient_step)
    ) / (2.0 * direction_step)
    fortml_gradient = np.array([
        float(record[f"gamma_likelihood_latent_gradient_{i}"])
        for i in range(1, 7)
    ] + [float(record["gamma_likelihood_shape_gradient"])])
    fortml_hvp = np.array([
        float(record[f"gamma_likelihood_latent_hvp_{i}"])
        for i in range(1, 7)
    ] + [float(record["gamma_likelihood_shape_hvp"])])
    value_error = abs(
        float(record["gamma_likelihood_value"])
        - oracle_value(observations, latents, weights, log_shape)
    )
    gradient_error = float(np.max(np.abs(fortml_gradient - gradient_oracle)))
    hvp_error = float(np.max(np.abs(fortml_hvp - hvp_oracle)))
    maximum_error = max(value_error, gradient_error, hvp_error)
    if value_error > 3.0e-13 or gradient_error > 1.0e-7 or hvp_error > 8.0e-5:
        raise RuntimeError(
            "Gamma likelihood oracle failed: "
            f"value={value_error:.3e}, gradient={gradient_error:.3e}, "
            f"hvp={hvp_error:.3e}"
        )
    fitted_log_shape = float(record["gamma_likelihood_fitted_log_shape"])
    scipy_reference = 3.059853981971437
    fit_error = abs(fitted_log_shape - scipy_reference)
    if fit_error > 2.0e-6:
        raise RuntimeError(f"Gamma shape fit error is {fit_error:.3e}")

    repetitions = int(record["gamma_likelihood_repeats"])
    started = time.perf_counter()
    for _ in range(repetitions):
        numpy_products(observations, latents, weights, log_shape, direction)
    numpy_seconds = time.perf_counter() - started
    fortml_seconds = float(record["gamma_likelihood_product_seconds"])
    fortml_per_operation = fortml_seconds / repetitions
    numpy_per_operation = numpy_seconds / repetitions
    speed_ratio = numpy_per_operation / fortml_per_operation
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_gamma_likelihood", "device": "cpu",
                    "n_samples": observations.size, "repetitions": repetitions})
        row.update(values)
        rows.append(row)

    add(phase="joint_products_correctness", backend="fortml", status="pass",
        metric="value_gradient_hvp_max_abs_error", value=maximum_error,
        max_abs_error=maximum_error,
        oracle="independent NumPy scalar density central differences",
        notes=(f"value={value_error:.3e}; gradient={gradient_error:.3e}; "
               f"hvp={hvp_error:.3e}"))
    add(phase="joint_products_performance", backend="fortml", status="pass",
        seconds_per_operation=fortml_per_operation,
        metric="numpy_to_fortml_time_ratio", value=speed_ratio,
        max_abs_error=maximum_error, oracle="NumPy/SciPy analytical products",
        notes="one value-gradient and one joint HVP per repetition")
    add(phase="joint_products_performance", backend="numpy_scipy", status="pass",
        seconds_per_operation=numpy_per_operation,
        metric="reference_seconds_per_operation", value=numpy_per_operation,
        max_abs_error=maximum_error, oracle="NumPy/SciPy analytical products",
        notes="Python loop invokes vectorized NumPy/SciPy products")
    add(phase="fortopt_shape_fit", backend="fortml", status="pass",
        repetitions=1,
        seconds_per_operation=float(record["gamma_likelihood_fit_seconds"]),
        metric="fitted_log_shape_abs_error", value=fit_error,
        max_abs_error=fit_error, oracle="SciPy bounded scalar minimization",
        notes=(f"log_shape={fitted_log_shape:.16e}; "
               f"gradient_norm={float(record['gamma_likelihood_fit_gradient_norm']):.3e}; "
               f"iterations={int(record['gamma_likelihood_fit_iterations'])}"))
    add(phase="device_contract", backend="fortml", device="cuda",
        status="unavailable", metric="resident_gamma_special_functions",
        value="nan", max_abs_error="nan",
        oracle="typed FORTNUM_NOT_IMPLEMENTED refusal",
        notes=("CUDA log-gamma, digamma, trigamma, and weighted reductions are "
               "not linked; no host fallback"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Weighted Gamma GP likelihood\n\n"
        "`bench_gp_gamma_likelihood.py` checks the positive-target Gamma density "
        "over the joint latent and transformed-shape coordinate space. An "
        "independent NumPy scalar density supplies central-difference gradient "
        "and directional-HVP oracles. The harness also checks the bounded "
        "FortOpt shape fit against SciPy.\n\n"
        "Run:\n\n"
        "```bash\n"
        "python -B scripts/bench_gp_gamma_likelihood.py --fortml ../fortml "
        "--output results/gp_gamma_likelihood.csv "
        "--report results/GP_GAMMA_LIKELIHOOD.md\n"
        "```\n\n"
        f"The maximum product error is `{maximum_error:.3e}`. The fitted "
        f"log shape differs from the SciPy optimum by `{fit_error:.3e}`. "
        f"FortML took `{fortml_per_operation:.3e}` seconds for one joint "
        f"value-gradient and HVP pair. NumPy/SciPy took "
        f"`{numpy_per_operation:.3e}` seconds in the Python loop, giving a "
        f"NumPy-to-FortML time ratio of `{speed_ratio:.3f}`. The source revision "
        f"is `{details['fortml_revision']}` and the benchmark revision is "
        f"`{details['benchmark_revision']}`. CUDA has a typed refusal until "
        "resident special functions and reductions are linked.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
