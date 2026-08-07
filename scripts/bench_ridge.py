#!/usr/bin/env python3
"""Correctness-gated weighted ridge regression benchmark.

The NumPy implementation is deliberately independent of FortML.  It uses the
closed-form weighted normal equations (with an unregularized intercept), then
checks vector and matrix prediction plus coefficient/input JVP and VJP
products.  FortML rows are retained only when a future release app exports
the complete arrays in the documented machine-readable protocol; a checkout
without that app receives explicit ``unavailable`` rows.
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


N_SAMPLES = 96
N_FEATURES = 6
N_OUTPUTS = 3
ALPHA = 0.37
REPETITIONS = 24
JVP_STEP = 1.0e-6
ORACLE_TOLERANCE = 2.0e-10

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_outputs", "alpha", "fit_intercept", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
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


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    generated_outputs = (output, root / "results" / "ridge.csv",
                         root / "results" / "cuda_adamw.csv")
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, generated_outputs),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "backend": "numpy_oracle", "device": "cpu", "status": "pass",
        "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_outputs": N_OUTPUTS, "alpha": ALPHA, "fit_intercept": "true",
        "repetitions": REPETITIONS, "oracle": "independent NumPy weighted ridge",
    })
    row.update(values)
    return row


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic, nondegenerate data and derivative fixtures."""
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    cols = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.071 * rows * cols) + 0.03 * np.cos(0.11 * rows + cols)
    weights = 0.4 + 0.06 * (1.0 + np.sin(0.17 * rows[:, 0]))
    latent = np.column_stack([
        0.7 + 0.8 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2],
        -0.3 + 0.4 * x[:, 2] + 0.9 * x[:, 3] - 0.25 * x[:, 4],
        0.2 - 0.5 * x[:, 1] + 0.3 * x[:, 4] + 0.6 * x[:, 5],
    ])
    y = latent + 0.02 * np.column_stack([
        np.sin(0.13 * rows[:, 0]), np.cos(0.09 * rows[:, 0]),
        np.sin(0.05 * rows[:, 0] + 0.3),
    ])
    x_dot = 0.07 * np.cos(0.037 * rows * (cols + 1.0))
    coefficient_dot = np.linspace(-0.17, 0.23, (N_FEATURES + 1) * N_OUTPUTS)
    u = np.column_stack([
        np.sin(0.041 * rows[:, 0]), np.cos(0.053 * rows[:, 0]),
        np.sin(0.067 * rows[:, 0] + 0.2),
    ])
    return x, y, weights, x_dot, coefficient_dot, u


def weighted_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                   alpha: float) -> np.ndarray:
    design = np.column_stack((np.ones(x.shape[0]), x))
    gram = design.T @ (weights[:, None] * design)
    gram[1:, 1:] += alpha * np.eye(x.shape[1])
    rhs = design.T @ (weights[:, None] * y)
    return np.linalg.solve(gram, rhs)


def evaluate_fixture() -> dict[str, np.ndarray]:
    x, y, weights, x_dot, coefficient_dot, u = fixture()
    coefficient = weighted_ridge(x, y, weights, ALPHA)
    design = np.column_stack((np.ones(N_SAMPLES), x))
    design_dot = np.column_stack((np.zeros(N_SAMPLES), x_dot))
    prediction = design @ coefficient
    coefficient_dot_matrix = coefficient_dot.reshape(coefficient.shape, order="F")
    prediction_dot = design_dot @ coefficient + design @ coefficient_dot_matrix
    theta_bar = (design.T @ u).reshape(-1, order="F")
    x_bar = u @ coefficient[1:, :].T

    # Independent behavioral checks: the JVP finite difference and VJP
    # adjoint identity must hold before any NumPy timing is retained.
    plus = (np.column_stack((np.ones(N_SAMPLES), x + JVP_STEP * x_dot)) @
            (coefficient + JVP_STEP * coefficient_dot_matrix))
    minus = (np.column_stack((np.ones(N_SAMPLES), x - JVP_STEP * x_dot)) @
             (coefficient - JVP_STEP * coefficient_dot_matrix))
    fd_error = float(np.max(np.abs(prediction_dot - (plus - minus) / (2.0 * JVP_STEP))))
    # The complete directional derivative includes the x contribution.  Keep
    # this expression separate so a shape/order mistake cannot hide in a
    # scalar checksum.
    adjoint_error = float(abs(np.sum(u * prediction_dot) -
                              (np.sum(theta_bar * coefficient_dot) +
                               np.sum(x_bar * x_dot))))
    if fd_error > 3.0e-9 or adjoint_error > 3.0e-12:
        raise RuntimeError(
            f"ridge derivative oracle failed: finite_difference={fd_error:.3e}, "
            f"adjoint={adjoint_error:.3e}"
        )
    return {
        "coefficient": coefficient,
        "prediction_matrix": prediction,
        "prediction_vector": prediction[:, 0],
        "prediction_jvp": prediction_dot,
        "prediction_vjp_theta": theta_bar,
        "prediction_vjp_x": x_bar,
        "fd_error": np.array([fd_error]),
        "adjoint_error": np.array([adjoint_error]),
    }


def timed(operation: Any, repetitions: int = REPETITIONS) -> tuple[Any, float]:
    started = time.perf_counter()
    value = None
    for _ in range(repetitions):
        value = operation()
    return value, (time.perf_counter() - started) / repetitions


def oracle_rows(details: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    x, y, weights, x_dot, coefficient_dot, u = fixture()
    values = evaluate_fixture()
    coefficient = values["coefficient"]
    design = np.column_stack((np.ones(N_SAMPLES), x))
    design_dot = np.column_stack((np.zeros(N_SAMPLES), x_dot))
    coefficient_dot_matrix = coefficient_dot.reshape(coefficient.shape, order="F")
    operations = {
        "fit_matrix": lambda: weighted_ridge(x, y, weights, ALPHA),
        "fit_vector": lambda: weighted_ridge(x, y[:, :1], weights, ALPHA),
        "predict_matrix": lambda: design @ coefficient,
        "predict_vector": lambda: (design @ coefficient)[:, 0],
        "predict_jvp": lambda: design_dot @ coefficient + design @ coefficient_dot_matrix,
        "predict_vjp_theta": lambda: (design.T @ u).reshape(-1, order="F"),
        "predict_vjp_x": lambda: u @ coefficient[1:, :].T,
    }
    rows: list[dict[str, Any]] = []
    expected = {
        "fit_matrix": values["coefficient"],
        "fit_vector": values["coefficient"][:, :1],
        "predict_matrix": values["prediction_matrix"],
        "predict_vector": values["prediction_vector"],
        "predict_jvp": values["prediction_jvp"],
        "predict_vjp_theta": values["prediction_vjp_theta"],
        "predict_vjp_x": values["prediction_vjp_x"],
    }
    for workload, operation in operations.items():
        actual, seconds = timed(operation)
        actual_array = np.asarray(actual)
        error = float(np.max(np.abs(actual_array - expected[workload])))
        if error > 1.0e-13:
            raise RuntimeError(f"NumPy {workload} self-check failed: {error:.3e}")
        rows.append(base(
            details, workload=workload,
            phase="fit" if workload.startswith("fit") else "predict",
            n_outputs=1 if workload in ("fit_vector", "predict_vector") else N_OUTPUTS,
            repetitions=REPETITIONS, seconds_per_operation=seconds,
            metric="l2_norm", value=float(np.linalg.norm(actual_array)),
            max_abs_error=error,
            notes=(f"shape={list(actual_array.shape)}; weighted closed form; "
                   f"derivative_fd_error={values['fd_error'][0]:.3e}; "
                   f"adjoint_error={values['adjoint_error'][0]:.3e}"),
        ))
    return rows, expected


def unavailable_rows(details: dict[str, str], reason: str) -> list[dict[str, Any]]:
    names = ("fit_matrix", "fit_vector", "predict_matrix", "predict_vector",
             "predict_jvp", "predict_vjp_theta", "predict_vjp_x")
    return [base(
        details, workload=name,
        phase="fit" if name.startswith("fit") else "predict",
        backend="fortml", device="cpu", status="unavailable",
        repetitions="", seconds_per_operation="", metric="l2_norm", value="",
        max_abs_error="", oracle="FortML complete-array release-app protocol",
        notes=reason,
    ) for name in names]


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               expected: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Run the reserved strict protocol if a release app is present.

    Each line is ``ridge,<workload>,<one-based-index>,<value>,<seconds>``.
    Every expected element must be present; a checksum-only app is rejected.
    """
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return unavailable_rows(details, f"release target source is absent: {source.name}")
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return unavailable_rows(details, "fo build failed; no FortML timing retained")
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                         env=environment, capture_output=True, text=True)
    if run.returncode != 0:
        return unavailable_rows(details, "release target execution failed")
    records: dict[str, dict[int, tuple[float, float]]] = {}
    for line in run.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 5 or fields[0] != "ridge":
            continue
        name = fields[1]
        if name not in expected:
            continue
        try:
            index, value, seconds = int(fields[2]), float(fields[3]), float(fields[4])
        except ValueError as error:
            raise RuntimeError(f"invalid FortML ridge protocol row: {line!r}") from error
        records.setdefault(name, {})[index] = (value, seconds)
    rows: list[dict[str, Any]] = []
    for name, target_array in expected.items():
        flat_expected = target_array.reshape(-1, order="F")
        record = records.get(name, {})
        if set(record) != set(range(1, flat_expected.size + 1)):
            raise RuntimeError(f"FortML ridge protocol omitted complete {name} array")
        actual = np.array([record[index][0] for index in range(1, flat_expected.size + 1)])
        error = float(np.max(np.abs(actual - flat_expected)))
        if error > ORACLE_TOLERANCE:
            raise RuntimeError(f"FortML ridge {name} oracle mismatch: {error:.3e}")
        seconds = float(np.median([record[index][1] for index in record]))
        rows.append(base(
            details, workload=name, phase="fit" if name.startswith("fit") else "predict",
            backend="fortml", status="pass", repetitions=REPETITIONS,
            seconds_per_operation=seconds, metric="l2_norm", value=float(np.linalg.norm(actual)),
            max_abs_error=error, oracle="FortML complete-array release-app protocol",
            notes=f"target={target}; entries={flat_expected.size}",
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/ridge.csv"))
    parser.add_argument("--target", default="fortml_bench_ridge_regression")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    rows, expected = oracle_rows(details)
    if args.skip_fortml:
        rows.extend(unavailable_rows(details, "--skip-fortml"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, expected))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
