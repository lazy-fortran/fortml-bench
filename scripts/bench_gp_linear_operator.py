#!/usr/bin/env python3
"""Correctness-gated benchmark for registered first-order GP operators."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


X = np.array([[-0.7], [-0.1], [0.45], [1.0]], dtype=np.float64)
Y = np.array([[0.6], [-0.2], [0.75], [0.35]], dtype=np.float64)
QUERY = np.array([[-0.5], [0.0], [0.6], [1.2]], dtype=np.float64)
TRAIN_WEIGHTS = np.array(
    [[1.0, 0.0, 0.7, 0.5], [0.0, 1.0, -0.4, 0.25]], dtype=np.float64
)
QUERY_WEIGHTS = TRAIN_WEIGHTS.copy()
DIRECTION = np.array(
    [[0.05, -0.07, 0.03, 0.02], [-0.04, 0.06, -0.02, 0.08]],
    dtype=np.float64,
)
MEAN_BAR = np.array([[0.17], [0.14], [0.11], [0.08]], dtype=np.float64)
VARIANCE_BAR = np.array([-0.06, -0.04, -0.02, 0.0], dtype=np.float64)
KERNEL_VARIANCE = 1.25
KERNEL_LENGTHSCALE = 0.82
NOISE = 0.06
JITTER = 1.0e-11
FD_STEP = 2.0e-6
REPETITIONS = 64
FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status",
    "n_train", "n_validation", "n_parameters", "evaluations", "repetitions",
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


def block(x1: float, w1: np.ndarray, x2: float, w2: np.ndarray) -> float:
    difference = x1 - x2
    inverse = 1.0 / KERNEL_LENGTHSCALE**2
    value = KERNEL_VARIANCE * np.exp(-0.5 * difference**2 * inverse)
    gradient_left = -value * difference * inverse
    gradient_right = -gradient_left
    mixed = value * (inverse - difference**2 * inverse**2)
    return float(
        w1[0] * w2[0] * value
        + w1[1] * w2[0] * gradient_left
        + w1[0] * w2[1] * gradient_right
        + w1[1] * w2[1] * mixed
    )


def covariance(x1: np.ndarray, weights1: np.ndarray,
              x2: np.ndarray, weights2: np.ndarray) -> np.ndarray:
    return np.array(
        [[block(float(left[0]), weights1[:, i], float(right[0]), weights2[:, j])
          for j, right in enumerate(x2)] for i, left in enumerate(x1)],
        dtype=np.float64,
    )


def predict(query_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = covariance(X, TRAIN_WEIGHTS, X, TRAIN_WEIGHTS)
    train[np.diag_indices_from(train)] += NOISE + JITTER
    cross = covariance(X, TRAIN_WEIGHTS, QUERY, query_weights)
    prior = covariance(QUERY, query_weights, QUERY, query_weights)
    alpha = np.linalg.solve(train, Y)
    mean = cross.T @ alpha
    work = np.linalg.solve(train, cross)
    variance = np.diag(prior) - np.sum(cross * work, axis=0)
    return mean, variance


def oracle() -> dict[str, np.ndarray | float]:
    mean, variance = predict(QUERY_WEIGHTS)
    plus_mean, plus_variance = predict(QUERY_WEIGHTS + FD_STEP * DIRECTION)
    minus_mean, minus_variance = predict(QUERY_WEIGHTS - FD_STEP * DIRECTION)
    mean_dot = (plus_mean - minus_mean) / (2.0 * FD_STEP)
    variance_dot = (plus_variance - minus_variance) / (2.0 * FD_STEP)
    mean_bar = MEAN_BAR
    lhs = float(np.sum(mean_bar * mean_dot) + np.dot(VARIANCE_BAR, variance_dot))
    return {
        "mean": mean[:, 0], "variance": variance,
        "mean_dot": mean_dot[:, 0], "variance_dot": variance_dot,
        "adjoint_error": 0.0, "adjoint_lhs": lhs,
    }


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "device": "cpu", "n_train": X.shape[0], "n_validation": QUERY.shape[0],
        "n_parameters": 0, "evaluations": 1, "repetitions": REPETITIONS,
    })
    row.update(values)
    return row


def parse_app(stdout: str) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], float] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0] == "operator_gp":
            if fields[1] == "cuda_status":
                values[(fields[1], 0)] = float(fields[2])
            elif fields[1] == "predict_seconds":
                values[(fields[1], 0)] = float(fields[2])
            elif fields[1] == "adjoint_error":
                values[(fields[1], 0)] = float(fields[2])
        elif len(fields) == 4 and fields[0] == "operator_gp":
            values[(fields[1], int(fields[2]))] = float(fields[3])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_linear_operator.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/GP_LINEAR_OPERATOR.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    expected = oracle()
    environment = os.environ.copy()
    environment.update({
        "FO_FC": environment.get("FO_FC", "gfortran"),
        "FO_SCAN_FALLBACK": environment.get("FO_SCAN_FALLBACK", "regex"),
        "OMP_NUM_THREADS": "1",
    })
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(f"FortML optimized build failed:\n{build.stderr[-2000:]}")
    run = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_gp_linear_operator"],
        cwd=fortml, env=environment, capture_output=True, text=True,
    )
    if run.returncode != 0:
        raise RuntimeError(f"FortML operator-GP app failed:\n{run.stderr[-2000:]}")
    actual = parse_app(run.stdout)
    rows: list[dict[str, Any]] = []
    oracle_name = "independent NumPy RBF operator covariance and central coefficient difference"
    for metric in ("mean", "variance", "mean_dot", "variance_dot"):
        values = np.asarray(expected[metric])
        for index, value in enumerate(values, start=1):
            rows.append(base(
                details, workload="gp_linear_operator", phase="oracle",
                variant="value_gradient_robin", backend="numpy_oracle", status="pass",
                metric=f"{metric}_{index}", value=float(value), max_abs_error=0.0,
                oracle=oracle_name, notes="1D RBF; coefficients=[value,d/dx]",
            ))
            key = (metric, index)
            if key not in actual:
                raise RuntimeError(f"FortML app omitted {key}")
            error = abs(actual[key] - float(value))
            tolerance = 2.0e-10 if metric in {"mean", "variance"} else 3.0e-7
            if error > tolerance:
                raise RuntimeError(f"FortML {metric}[{index}] mismatch: {error:.3e}")
            rows.append(base(
                details, workload="gp_linear_operator", phase="prediction" if metric in {"mean", "variance"} else "operator_coefficient_jvp",
                variant="value_gradient_robin", backend="fortml", status="pass",
                seconds_per_operation=actual.get(("predict_seconds", 0), ""),
                metric=f"{metric}_{index}", value=actual[key], max_abs_error=error,
                oracle=oracle_name, notes="exact dense CPU operator covariance",
            ))
    adjoint_error = actual[("adjoint_error", 0)]
    if adjoint_error > 3.0e-10:
        raise RuntimeError(f"operator JVP/VJP adjoint mismatch: {adjoint_error:.3e}")
    rows.append(base(
        details, workload="gp_linear_operator", phase="operator_coefficient_vjp",
        variant="value_gradient_robin", backend="fortml", status="pass",
        metric="jvp_vjp_adjoint_error", value=adjoint_error, max_abs_error=adjoint_error,
        oracle="operator coefficient adjoint identity", notes="mean/variance cotangent pullback",
    ))
    cuda_status = int(actual[("cuda_status", 0)])
    if cuda_status <= 0:
        raise RuntimeError("FortML operator-GP CUDA request was not refused")
    rows.append(base(
        details, workload="gp_linear_operator", phase="device_contract",
        variant="value_gradient_robin", backend="fortml", device="cuda", status="refused",
        metric="resident_operator_covariance", oracle="FortML typed device contract",
        notes=f"no host fallback; FORTNUM_NOT_IMPLEMENTED status={cuda_status}",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    predict_seconds = actual[("predict_seconds", 0)]
    report.write_text(
        "# Registered linear-operator GP\n\n"
        "This lane checks a named first-order operator registry with columns "
        "`[value, d/dx]`. Three mixed operator rows (value, gradient, and two "
        "Robin combinations) are fitted and queried through an exact dense RBF GP.\n\n"
        "The independent NumPy oracle assembles value, gradient, and mixed-Hessian "
        "covariance blocks directly, solves the noisy dense system, and central-"
        "differences query operator coefficients for JVPs. The release Fortran "
        "test additionally checks the JVP/VJP adjoint identity. CUDA is an explicit "
        "typed refusal until a resident operator covariance graph is linked.\n\n"
        f"Maximum prediction error: `{max(float(row['max_abs_error']) for row in rows if row['status'] == 'pass'):.3e}`. "
        f"CPU prediction time: `{predict_seconds:.3e}` seconds per query batch.\n\n"
        f"FortML revision: `{details['fortml_revision']}`. Benchmark revision: "
        f"`{details['benchmark_revision']}`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
