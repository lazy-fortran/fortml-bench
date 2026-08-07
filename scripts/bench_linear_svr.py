#!/usr/bin/env python3
"""Correctness-gated benchmark for weighted dense linear SVR.

The NumPy/SciPy primal squared epsilon-insensitive solve is an independent
oracle.  FortML rows are retained only after targets, weights, and every
prediction agree.  CUDA remains an explicit typed capability refusal until a
resident SVR kernel is linked.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scipy
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    scipy = None
    minimize = None


N_SAMPLES, N_FEATURES = 192, 4
L2, EPSILON = 5.0e-2, 8.0e-2
PREDICTION_REPETITIONS = 128
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "seconds_per_operation", "accuracy", "max_abs_error",
    "oracle", "python_version", "numpy_version", "scipy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        if name not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * phase[:, None] + 0.071 * columns)
    x += 0.12 * np.cos(0.013 * phase[:, None] * columns)
    truth = (0.8 * x[:, 0] - 0.45 * x[:, 1] + 0.25 * x[:, 2] -
             0.15 * x[:, 3] + 0.10 * np.sin(0.11 * phase))
    targets = truth + 0.08 * np.cos(0.07 * phase)
    weights = 0.75 + 0.5 * (np.mod(np.arange(1, N_SAMPLES + 1), 9) / 8.0)
    return x, targets, weights


def objective_and_gradient(theta: np.ndarray, x: np.ndarray, targets: np.ndarray,
                           weights: np.ndarray) -> tuple[float, np.ndarray]:
    beta, intercept = theta[:N_FEATURES], theta[N_FEATURES]
    residual = x @ beta + intercept - targets
    excess = np.maximum(0.0, np.abs(residual) - EPSILON)
    mass = float(weights.sum())
    value = float(np.dot(weights, excess * excess) / mass +
                  0.5 * L2 * np.dot(beta, beta))
    residual_gradient = 2.0 * excess * np.sign(residual)
    gradient = np.empty_like(theta)
    gradient[:N_FEATURES] = x.T @ (weights * residual_gradient) / mass + L2 * beta
    gradient[N_FEATURES] = np.dot(weights, residual_gradient) / mass
    return value, gradient


def oracle(x: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, float]:
    if minimize is None:
        raise RuntimeError("SciPy is required for the independent linear-SVR oracle")
    started = time.perf_counter()
    result = minimize(
        lambda theta: objective_and_gradient(theta, x, targets, weights),
        np.zeros(N_FEATURES + 1, dtype=np.float64), method="L-BFGS-B", jac=True,
        options={"maxiter": 1500, "ftol": 1.0e-15, "gtol": 1.0e-9, "maxls": 50},
    )
    fit_seconds = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(f"independent SVR oracle failed: {result.message}")
    theta = result.x
    prediction = x @ theta[:N_FEATURES] + theta[N_FEATURES]
    return theta, prediction, fit_seconds


def parse_fortran(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    prediction = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    weights = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    parameters = np.full(N_FEATURES + 1, np.nan, dtype=np.float64)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            quantity = record["quantity"]
            row = int(record["row"]) - 1
            value = float(record["value"])
            if quantity == "target":
                targets[row] = value
            elif quantity == "prediction":
                prediction[row] = value
            elif quantity == "weight":
                weights[row] = value
            elif quantity == "parameter":
                parameters[row] = value
            else:
                raise RuntimeError(f"unknown FortML quantity {quantity!r}")
    if (np.isnan(targets).any() or np.isnan(prediction).any() or
            np.isnan(weights).any() or np.isnan(parameters).any()):
        raise RuntimeError("FortML omitted a linear-SVR output")
    return targets, prediction, weights, parameters


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": "unavailable" if scipy is None else scipy.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }


def make_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "workload": "linear_svr_regression", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "seconds_per_operation": "", "accuracy": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    }
    row.update(details)
    row.update(values)
    return row


def parse_timing(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 4 and fields[0] in {"linear_svr_fit", "linear_svr_predict"}:
            values[fields[0]] = float(fields[-1])
    return values


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    x, targets, weights = fixture()
    theta, expected, fit_seconds = oracle(x, targets, weights)
    started = time.perf_counter()
    for _ in range(PREDICTION_REPETITIONS):
        x @ theta[:N_FEATURES] + theta[N_FEATURES]
    predict_seconds = (time.perf_counter() - started) / PREDICTION_REPETITIONS
    return [
        make_row(details, phase="fit", backend="numpy_oracle", status="pass",
                 seconds_per_operation=fit_seconds, accuracy="", max_abs_error=0.0,
                 oracle="independent SciPy L-BFGS-B weighted squared epsilon solve"),
        make_row(details, phase="predict", backend="numpy_oracle", status="pass",
                 seconds_per_operation=predict_seconds, accuracy="", max_abs_error=0.0,
                 oracle="NumPy affine prediction from independent fitted weights"),
    ]


def run_fortml(root: Path, fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                   env=environment, check=True)
    x, expected_targets, expected_weights = fixture()
    theta, expected_prediction, _ = oracle(x, expected_targets, expected_weights)
    with tempfile.TemporaryDirectory(dir="/mnt/storage/code/lazy-fortran/fortml/build") as directory:
        output = Path(directory) / "linear_svr_oracle.csv"
        environment["FORTML_BENCH_LINEAR_SVR_ORACLE"] = str(output)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_linear_svr"],
            cwd=fortml, env=environment, capture_output=True, text=True, check=True,
        )
        targets, prediction, weights, parameters = parse_fortran(output)
    if not np.allclose(targets, expected_targets, rtol=0.0, atol=2.0e-14):
        raise RuntimeError("FortML linear-SVR fixture targets differ from NumPy")
    if not np.allclose(weights, expected_weights, rtol=0.0, atol=2.0e-14):
        raise RuntimeError("FortML linear-SVR fixture weights differ from NumPy")
    expected_parameters = theta
    error = float(np.max(np.abs(prediction - expected_prediction)))
    parameter_error = float(np.max(np.abs(parameters - expected_parameters)))
    if max(error, parameter_error) > 2.0e-4:
        raise RuntimeError(
            f"FortML linear-SVR oracle mismatch prediction={error:.3e} "
            f"parameters={parameter_error:.3e}")
    timing = parse_timing(completed.stdout)
    return [
        make_row(details, phase="fit", backend="fortml_cpu", status="pass",
                 seconds_per_operation=timing.get("linear_svr_fit", ""),
                 max_abs_error=max(error, parameter_error),
                 oracle="NumPy/SciPy complete target/weight/prediction oracle",
                 notes="FortOpt L-BFGS-B weighted squared epsilon-insensitive loss"),
        make_row(details, phase="predict", backend="fortml_cpu", status="pass",
                 seconds_per_operation=timing.get("linear_svr_predict", ""),
                 max_abs_error=error, oracle="NumPy affine prediction oracle",
                 notes="fixed fitted affine SVR prediction"),
        make_row(details, phase="predict", backend="fortml_cuda", device="cuda",
                 status="unavailable", oracle="typed_device_contract",
                 notes="no resident linear-SVR CUDA kernel; FORTNUM_NOT_IMPLEMENTED"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/linear_svr.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = metadata(root, args.fortml.resolve(), args.output.resolve())
    rows = run_numpy(details) + run_fortml(root, args.fortml.resolve(), details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
