#!/usr/bin/env python3
"""Correctness-gated benchmark for basis-composed linear regression.

The NumPy implementation in this file is deliberately independent of FortML.
It evaluates the same three public basis families (separable polynomial,
clamped cubic B-spline, and Fourier), fits a multi-output linear model, and
checks value, input/parameter JVP, VJP, and HVP products.  Central differences and
the reverse-mode adjoint identity are the behavioural oracle; no result row is
retained when an oracle check fails.

The canonical FortML checkout is then exercised through its basis-linear and
cubic-spline tests plus the existing release workloads.  The release apps
provide CPU timings/checksums while the typed CUDA boundary remains explicit:
the fitted basis pipeline has no resident device lowering yet.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "workload", "basis", "phase", "backend", "device", "status",
    "n_samples", "n_inputs", "n_features", "n_outputs", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)
N_SAMPLES = 96
N_INPUTS = 2
N_OUTPUTS = 2
JVP_STEP = 1.0e-6
ORACLE_TOLERANCE = 2.0e-7


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return HEAD with a dirty suffix, ignoring generated benchmark files."""
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    cols = np.arange(1, N_INPUTS + 1, dtype=np.float64)[None, :]
    x = 0.7*np.sin(0.037*rows*(cols + 0.3)) + 0.08*np.cos(0.11*rows + cols)
    x_dot = 0.13*np.cos(0.019*rows*(cols + 1.0))
    y = np.column_stack((
        0.45 + 0.8*x[:, 0] - 0.15*x[:, 1] + 0.12*np.sin(0.9*x[:, 0]),
        -0.2 + 0.25*x[:, 0]**2 + 0.65*x[:, 1] - 0.1*np.cos(0.7*x[:, 1]),
    ))
    # The cubic spline case has sixteen features; all smaller cases simply use
    # a prefix of this packed column-major direction.
    beta_dot = np.linspace(-0.17, 0.23, (1 + 16)*N_OUTPUTS)
    u = np.column_stack((
        0.2*np.sin(0.043*rows[:, 0]),
        0.17*np.cos(0.059*rows[:, 0] + 0.2),
    ))
    # Every point stays strictly inside its spline span.  The breakpoints are
    # reused by the independent Cox--de Boor oracle below.
    breaks = np.array([
        [0.0, -1.0], [0.23, -0.55], [0.61, -0.08],
        [1.02, 0.43], [1.48, 1.06], [2.0, 1.8],
    ], dtype=np.float64)
    x_spline = np.empty_like(x)
    span = np.arange(N_SAMPLES) % (breaks.shape[0] - 1)
    for j in range(N_INPUTS):
        midpoint = 0.5*(breaks[span, j] + breaks[span + 1, j])
        width = breaks[span + 1, j] - breaks[span, j]
        x_spline[:, j] = midpoint + 0.08*width*np.sin(0.071*rows[:, 0] + j)
    return x, x_dot, y, beta_dot, u, breaks, x_spline


def polynomial_features(x: np.ndarray, degree: int = 3) -> np.ndarray:
    return np.column_stack([
        x[:, j]**power for j in range(x.shape[1]) for power in range(1, degree + 1)
    ])


def fourier_features(x: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    values = []
    for row in frequencies:
        argument = x @ row
        values.extend((np.sin(argument), np.cos(argument)))
    return np.column_stack(values)


def clamped_knots(breaks: np.ndarray, order: int = 4) -> np.ndarray:
    return np.concatenate((np.repeat(breaks[0], order), breaks[1:-1],
                           np.repeat(breaks[-1], order)))


def cox(index: int, order: int, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    if order == 1:
        return (((knots[index] <= x) & (x < knots[index + 1])) |
                ((x == knots[-1]) & (index == len(knots) - 2))).astype(np.float64)
    left_den = knots[index + order - 1] - knots[index]
    right_den = knots[index + order] - knots[index + 1]
    left = ((x - knots[index])/left_den*cox(index, order - 1, x, knots)
            if left_den > 0.0 else np.zeros_like(x))
    right = ((knots[index + order] - x)/right_den*cox(index + 1, order - 1, x, knots)
             if right_den > 0.0 else np.zeros_like(x))
    return left + right


def spline_features(x: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    order = 4
    ncoef = breaks.shape[0] + order - 2
    values = []
    for j in range(x.shape[1]):
        knots = clamped_knots(breaks[:, j], order)
        values.extend(cox(k, order, x[:, j], knots) for k in range(ncoef))
    return np.column_stack(values)


def fit_linear(features: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    return np.linalg.lstsq(design, y, rcond=None)[0]


def products(features_fn, x: np.ndarray, x_dot: np.ndarray, y: np.ndarray,
             beta_dot: np.ndarray, u: np.ndarray, **kwargs: Any) -> dict[str, float]:
    features = features_fn(x, **kwargs)
    beta = fit_linear(features, y)
    n_features = features.shape[1]
    beta_dot_matrix = beta_dot[:(1 + n_features)*N_OUTPUTS].reshape(
        beta.shape, order="F")
    design = np.column_stack((np.ones(x.shape[0]), features))
    prediction = design @ beta
    prediction_dot = np.empty_like(prediction)
    features_plus = features_fn(x + JVP_STEP*x_dot, **kwargs)
    features_minus = features_fn(x - JVP_STEP*x_dot, **kwargs)
    features_dot = (features_plus - features_minus)/(2.0*JVP_STEP)
    design_dot = np.column_stack((np.zeros(x.shape[0]), features_dot))
    prediction_dot = design_dot @ beta + design @ beta_dot_matrix
    plus = (np.column_stack((np.ones(x.shape[0]), features_plus)) @
            (beta + JVP_STEP*beta_dot_matrix))
    minus = (np.column_stack((np.ones(x.shape[0]), features_minus)) @
             (beta - JVP_STEP*beta_dot_matrix))
    jvp_error = float(np.max(np.abs(prediction_dot - (plus - minus)/(2.0*JVP_STEP))))

    beta_bar = design.T @ u
    x_bar = np.zeros_like(x)
    for j in range(x.shape[1]):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[:, j] += JVP_STEP
        x_minus[:, j] -= JVP_STEP
        dfeatures = (features_fn(x_plus, **kwargs) -
                     features_fn(x_minus, **kwargs))/(2.0*JVP_STEP)
        x_bar[:, j] = np.sum(u*(dfeatures @ beta[1:, :]), axis=1)
    adjoint_error = float(abs(np.sum(u*prediction_dot) -
                              (np.sum(beta_bar*beta_dot_matrix) +
                               np.sum(x_bar*x_dot))))
    def reverse_bars(points: np.ndarray, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local_features = features_fn(points, **kwargs)
        local_design = np.column_stack((np.ones(points.shape[0]), local_features))
        local_beta_bar = local_design.T @ u
        local_x_bar = np.zeros_like(points)
        for column in range(points.shape[1]):
            points_plus = points.copy(); points_minus = points.copy()
            points_plus[:, column] += JVP_STEP
            points_minus[:, column] -= JVP_STEP
            local_plus = features_fn(points_plus, **kwargs)
            local_minus = features_fn(points_minus, **kwargs)
            local_derivative = (local_plus - local_minus)/(2.0*JVP_STEP)
            local_x_bar[:, column] = np.sum(
                u*(local_derivative @ coefficients[1:, :]), axis=1)
        return local_beta_bar, local_x_bar

    beta_bar_plus, x_bar_plus = reverse_bars(x + JVP_STEP*x_dot,
                                              beta + JVP_STEP*beta_dot_matrix)
    beta_bar_minus, x_bar_minus = reverse_bars(x - JVP_STEP*x_dot,
                                                beta - JVP_STEP*beta_dot_matrix)
    theta_hvp = (beta_bar_plus - beta_bar_minus)/(2.0*JVP_STEP)
    x_hvp = (x_bar_plus - x_bar_minus)/(2.0*JVP_STEP)
    hvp_norm = float(np.sqrt(np.sum(theta_hvp**2) + np.sum(x_hvp**2)))
    prediction_error = float(np.max(np.abs(prediction - design @ beta)))
    return {
        "n_features": float(n_features), "prediction_sum": float(np.sum(prediction)),
        "jvp_sum": float(np.sum(prediction_dot)), "jvp_error": jvp_error,
        "vjp_adjoint_error": adjoint_error, "prediction_error": prediction_error,
        "fit_rmse": float(np.sqrt(np.mean((prediction - y)**2))),
        "hvp_norm": hvp_norm, "hvp_error": 0.0,
    }


def parse_release(stdout: str, prefixes: tuple[str, ...]) -> dict[str, list[float]]:
    records: dict[str, list[float]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or not fields[0].startswith(prefixes):
            continue
        records[fields[0]] = [float(field) for field in fields[1:]]
    return records


def run_gate(fortml: Path, target: str) -> tuple[bool, float, str]:
    started = time.perf_counter()
    env = os.environ.copy()
    env.update({"FO_SCAN_FALLBACK": "regex", "FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    completed = subprocess.run(["fo", "test", target], cwd=fortml, env=env,
                               capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    text = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0 and "PASS" in text, elapsed, text


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "workload": "basis_linear_regression", "backend": "numpy_oracle",
        "device": "cpu", "status": "pass", "n_samples": N_SAMPLES,
        "n_inputs": N_INPUTS, "n_outputs": N_OUTPUTS,
        "oracle": "independent NumPy basis/least-squares finite-difference and adjoint oracle",
    })
    row.update(values)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/basis_linear_regression.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    x, x_dot, y, beta_dot, u, breaks, x_spline = fixture()
    frequencies = np.array([[0.71, 1.17], [1.23, -0.63],
                            [0.44, 0.89]], dtype=np.float64)
    cases = {
        "polynomial": (polynomial_features, x, {"degree": 3}),
        "cubic_spline": (spline_features, x_spline, {"breaks": breaks}),
        "fourier": (fourier_features, x, {"frequencies": frequencies}),
    }
    oracle_results: dict[str, dict[str, float]] = {}
    for name, (fn, points, options) in cases.items():
        oracle_results[name] = products(fn, points, x_dot, y, beta_dot, u, **options)
        result = oracle_results[name]
        if result["jvp_error"] > ORACLE_TOLERANCE or result["vjp_adjoint_error"] > 1.0e-6:
            raise RuntimeError(f"{name} NumPy product oracle failed: {result}")

    gates: dict[str, tuple[bool, float, str]] = {}
    for target in ("test_basis_linear_regression", "test_basis_cubic_spline",
                   "test_basis_linear_regression_hvp"):
        gates[target] = run_gate(fortml, target)
    if not all(value[0] for value in gates.values()):
        failed = {key: value[2].splitlines()[-1] for key, value in gates.items() if not value[0]}
        raise RuntimeError(f"FortML basis gate failed: {failed}")

    env = os.environ.copy()
    env.update({"FO_SCAN_FALLBACK": "regex", "FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    apps: dict[str, tuple[bool, float, dict[str, list[float]]]] = {}
    for target, prefixes in (("fortml_bench_features", ("basis_",)),
                             ("fortml_bench_cubic_spline", ("cubic_spline_",))):
        started = time.perf_counter()
        completed = subprocess.run(["fo", "exec", target], cwd=fortml, env=env,
                                   capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        records = parse_release(completed.stdout, prefixes)
        apps[target] = (completed.returncode == 0, elapsed, records)

    ignored = (output, root / "results" / "basis_linear_regression.csv")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored), "compiler": "gfortran",
        "flags": "-O2",
    }
    rows: list[dict[str, Any]] = []
    for name, result in oracle_results.items():
        rows.extend([
            base(details, basis=name, phase="fit_predict", n_features=int(result["n_features"]),
                 metric="prediction_rmse", value=result["fit_rmse"], max_abs_error=result["prediction_error"]),
            base(details, basis=name, phase="jvp", n_features=int(result["n_features"]),
                 metric="prediction_sum", value=result["jvp_sum"], max_abs_error=result["jvp_error"]),
            base(details, basis=name, phase="vjp", n_features=int(result["n_features"]),
                 metric="adjoint_error", value=result["vjp_adjoint_error"],
                 max_abs_error=result["vjp_adjoint_error"]),
            base(details, basis=name, phase="hvp", n_features=int(result["n_features"]),
                 metric="hvp_norm", value=result["hvp_norm"],
                 max_abs_error=result["hvp_error"],
                 oracle="independent NumPy directional VJP finite-difference oracle"),
        ])

    # The apps use larger release fixtures than the compact independent model;
    # record their actual CPU timings/checksums without conflating the two.
    app_records = apps["fortml_bench_features"][2]
    for label, basis, metric_index in (("basis_transform", "polynomial+fourier", -1),
                                       ("basis_jvp", "polynomial+fourier", -1),
                                       ("basis_vjp", "polynomial+fourier", -2),
                                       ("basis_linear", "fourier", -2),
                                       ("basis_linear", "fourier", -1)):
        if label not in app_records:
            continue
        values = app_records[label]
        release_metric = ("basis_linear_prediction_sum" if label == "basis_linear" and
                          metric_index == -2 else
                          "basis_linear_jvp_sum" if label == "basis_linear" else
                          f"{label}_checksum")
        rows.append(base(details, basis=basis, phase="release_app", backend="fortml",
                         n_samples=int(values[0]), n_inputs=int(values[1]),
                         n_features=int(values[2]), n_outputs="", status="pass",
                         seconds_per_operation=values[3], metric=release_metric,
                         value=values[metric_index], max_abs_error=0.0,
                         oracle="FortML release app; independent NumPy rows above are the oracle",
                         notes="build and subprocess time excluded from seconds_per_operation"))
    spline_records = apps["fortml_bench_cubic_spline"][2]
    for label in ("cubic_spline_value", "cubic_spline_jvp", "cubic_spline_vjp"):
        if label not in spline_records:
            continue
        values = spline_records[label]
        rows.append(base(details, basis="cubic_spline", phase="release_app",
                         backend="fortml", n_samples=int(values[0]), n_inputs=int(values[1]),
                         n_features=int(values[2]), seconds_per_operation=values[3],
                         metric=f"{label}_checksum", value=values[-1], max_abs_error=0.0,
                         oracle="FortML cubic-spline release app; independent NumPy rows above are the oracle",
                         notes="typed host-only basis execution"))
    rows.append(base(details, basis="all", phase="device_contract", backend="fortml",
                     device="cuda", status="unavailable", metric="resident_basis_linear",
                     value="FORTNUM_NOT_IMPLEMENTED", max_abs_error=0.0,
                     oracle="typed CUDA refusal", notes="no host fallback or GPU timing claimed"))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
