#!/usr/bin/env python3
"""Correctness-gated binary GP likelihood and derivative benchmark.

The NumPy implementation is independent of FortML and covers the two
likelihoods used by the Laplace classifier.  It checks scalar values against
hand-written stable formulae, a central directional difference, and the VJP
adjoint identity before retaining timings.  A future complete-array release
app can be added without relabelling this oracle as a FortML or GPU timing.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


N = 4096
REPETITIONS = 48
STEP = 1.0e-4
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
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
    return head + ("+dirty" if dirty else "")


def stable_sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def normal_cdf(value: np.ndarray) -> np.ndarray:
    return 0.5 * np.vectorize(math.erfc, otypes=[float])(-value / math.sqrt(2.0))


def probit_logcdf(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value)
    direct = value > -8.0
    result[direct] = np.log(np.maximum(normal_cdf(value[direct]), np.finfo(float).tiny))
    tail = value[~direct]
    inverse_square = 1.0 / (tail * tail)
    correction = 1.0 - inverse_square + 3.0 * inverse_square * inverse_square
    result[~direct] = (-0.5 * tail * tail - np.log(-tail) -
                       0.9189385332046727 + np.log(np.maximum(correction, np.finfo(float).tiny)))
    return result


def likelihood(eta: np.ndarray, probit: bool) -> tuple[np.ndarray, np.ndarray]:
    if probit:
        log_value = probit_logcdf(eta)
        probability = normal_cdf(eta)
        density = np.exp(-0.5 * eta * eta) / math.sqrt(2.0 * math.pi)
        ratio = density / np.maximum(probability, 1.0e-14)
        negative_tail = probability <= 1.0e-14
        ratio[negative_tail] = np.maximum(1.0, -eta[negative_tail]) + \
            1.0 / np.maximum(1.0, -eta[negative_tail])
        return log_value, ratio
    probability = stable_sigmoid(eta)
    log_value = np.where(eta >= 0.0, -np.log1p(np.exp(-eta)),
                         eta - np.log1p(np.exp(eta)))
    return log_value, 1.0 - probability


def fixture() -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(1, N + 1, dtype=np.float64)
    eta = 2.1 * np.sin(0.013 * index) + 0.7 * np.cos(0.031 * index)
    tangent = 0.3 * np.cos(0.017 * index) - 0.2 * np.sin(0.023 * index)
    return eta, tangent


def evaluate(eta: np.ndarray, tangent: np.ndarray, probit: bool) -> dict[str, float]:
    log_value, derivative = likelihood(eta, probit)
    value = float(np.sum(log_value))
    value_dot = float(np.dot(derivative, tangent))
    eta_plus = eta + STEP * tangent
    eta_minus = eta - STEP * tangent
    value_plus = float(np.sum(likelihood(eta_plus, probit)[0]))
    value_minus = float(np.sum(likelihood(eta_minus, probit)[0]))
    finite_difference_error = abs(value_dot - (value_plus - value_minus) / (2.0 * STEP))
    eta_bar = derivative * 1.7
    adjoint_error = abs(float(np.dot(eta_bar, tangent)) - 1.7 * value_dot)
    if finite_difference_error > (2.0e-8 if probit else 2.0e-9) or adjoint_error > 2.0e-11:
        raise RuntimeError(
            f"GP likelihood oracle failed: probit={probit}, "
            f"fd={finite_difference_error:.3e}, adjoint={adjoint_error:.3e}"
        )
    return {
        "value": value, "jvp": value_dot, "vjp_norm": float(np.linalg.norm(eta_bar)),
        "finite_difference_error": finite_difference_error,
        "adjoint_error": adjoint_error,
    }


def timed(operation: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = None
    for _ in range(REPETITIONS):
        result = operation()
    return result, (time.perf_counter() - started) / REPETITIONS


def details(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran",
        "flags": "-O3",
    }


def row(metadata: dict[str, str], workload: str, phase: str, status: str,
        value: Any, seconds: Any, error: Any, notes: str,
        backend: str = "numpy_oracle", device: str = "cpu") -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update({
        "workload": workload, "phase": phase, "backend": backend,
        "device": device, "status": status, "n_samples": N,
        "repetitions": REPETITIONS, "seconds_per_operation": seconds,
        "metric": "scalar_or_norm", "value": value, "max_abs_error": error,
        "oracle": "independent NumPy signed-margin likelihood", "notes": notes,
    })
    return result


def run_fortml(fortml: Path, target: str, metadata: dict[str, str],
               expected: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Run the strict scalar release protocol when the app is available."""
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return [row(
            metadata, f"gp_likelihood_{name}", operation, "unavailable", "", "", "",
            f"release target source is absent: {source.name}",
            backend="fortml", device="cpu",
        ) for name in expected for operation in expected[name]]
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                        "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return [row(
            metadata, f"gp_likelihood_{name}", operation, "unavailable", "", "", "",
            "FortML release target did not build; no timing retained",
            backend="fortml", device="cpu",
        ) for name in expected for operation in expected[name]]
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                         env=environment, capture_output=True, text=True)
    if run.returncode != 0:
        return [row(
            metadata, f"gp_likelihood_{name}", operation, "unavailable", "", "", "",
            "FortML release target failed; no timing retained",
            backend="fortml", device="cpu",
        ) for name in expected for operation in expected[name]]
    records: dict[tuple[str, str], tuple[float, float]] = {}
    for line in run.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5 or fields[0] != "gp_likelihood":
            continue
        name, operation = fields[1], fields[2]
        if name not in expected or operation not in expected[name]:
            continue
        try:
            seconds, value = float(fields[3]), float(fields[4])
        except ValueError as error:
            raise RuntimeError(f"invalid FortML GP likelihood row: {line!r}") from error
        records[(name, operation)] = (seconds, value)
    rows: list[dict[str, Any]] = []
    for name, operations in expected.items():
        for operation, target_value in operations.items():
            if (name, operation) not in records:
                raise RuntimeError(
                    f"FortML GP likelihood protocol omitted {name}/{operation}"
                )
            seconds, actual = records[(name, operation)]
            error = abs(actual - target_value)
            if error > 2.0e-9:
                raise RuntimeError(
                    f"FortML GP likelihood oracle mismatch for {name}/{operation}: "
                    f"{error:.3e}"
                )
            rows.append(row(
                metadata, f"gp_likelihood_{name}", operation, "pass", actual,
                seconds, error,
                "strict scalar FortML release-app protocol; host execution",
                backend="fortml", device="cpu",
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_likelihood.csv"))
    parser.add_argument("--target", default="fortml_bench_gp_classification_likelihood")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    metadata = details(root, args.fortml.resolve(), output)
    eta, tangent = fixture()
    rows: list[dict[str, Any]] = []
    expected_fortran: dict[str, dict[str, float]] = {}
    for name, probit in (("logistic", False), ("probit", True)):
        values = evaluate(eta, tangent, probit)
        operations = {
            "value": lambda: np.sum(likelihood(eta, probit)[0]),
            "jvp": lambda: np.dot(likelihood(eta, probit)[1], tangent),
            "vjp": lambda: likelihood(eta, probit)[1] * 1.7,
        }
        expected = {
            "value": values["value"], "jvp": values["jvp"],
            "vjp": values["vjp_norm"],
        }
        expected_fortran[name] = expected
        for operation, function in operations.items():
            actual, seconds = timed(function)
            value = float(actual) if operation != "vjp" else float(np.linalg.norm(actual))
            error = abs(value - expected[operation])
            if error > 1.0e-12:
                raise RuntimeError(f"NumPy {name} {operation} self-check failed: {error:.3e}")
            rows.append(row(
                metadata, f"gp_likelihood_{name}", operation, "pass", value,
                seconds, error,
                f"fd_error={values['finite_difference_error']:.3e}; "
                f"adjoint_error={values['adjoint_error']:.3e}",
            ))
    rows.extend(run_fortml(args.fortml.resolve(), args.target, metadata,
                           expected_fortran))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
