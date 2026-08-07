#!/usr/bin/env python3
"""Correctness-gated weighted elastic-net benchmark.

The NumPy implementation independently solves the weighted multi-output
elastic-net objective with coordinate descent.  It checks every coefficient,
prediction, packed-parameter JVP/VJP, and input JVP/VJP value exported by the
FortML release app before retaining timing rows.  The fit-time active-set
decisions are intentionally not differentiated: derivative products use the
fixed fitted coefficient state exposed by FortML.
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
ALPHA = 0.21
L1_RATIO = 0.43
REPETITIONS = 24
ORACLE_TOLERANCE = 5.0e-10
JVP_STEP = 1.0e-6

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_outputs", "alpha", "l1_ratio", "fit_intercept",
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


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "backend": "numpy_oracle", "device": "cpu", "status": "pass",
        "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_outputs": N_OUTPUTS, "alpha": ALPHA, "l1_ratio": L1_RATIO,
        "fit_intercept": "true", "repetitions": REPETITIONS,
        "oracle": "independent NumPy weighted elastic-net coordinate descent",
    })
    result.update(values)
    return result


def fixture() -> tuple[np.ndarray, ...]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.071 * rows * columns) + 0.03 * np.cos(0.11 * rows + columns)
    weights = 0.4 + 0.06 * (1.0 + np.sin(0.17 * rows[:, 0]))
    latent = np.column_stack((
        0.7 + 0.8 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2],
        -0.3 + 0.4 * x[:, 2] + 0.9 * x[:, 3] - 0.25 * x[:, 4],
        0.2 - 0.5 * x[:, 1] + 0.3 * x[:, 4] + 0.6 * x[:, 5],
    ))
    y = latent + 0.02 * np.column_stack((
        np.sin(0.13 * rows[:, 0]),
        np.cos(0.09 * rows[:, 0]),
        np.sin(0.05 * rows[:, 0] + 0.3),
    ))
    x_dot = 0.07 * np.cos(0.037 * rows * (columns + 1.0))
    theta_dot = np.linspace(-0.17, 0.23, (N_FEATURES + 1) * N_OUTPUTS)
    u = np.column_stack((
        np.sin(0.041 * rows[:, 0]),
        np.cos(0.053 * rows[:, 0]),
        np.sin(0.067 * rows[:, 0] + 0.2),
    ))
    return x, y, weights, x_dot, theta_dot, u


def soft_threshold(value: float, threshold: float) -> float:
    return float(np.sign(value) * max(abs(value) - threshold, 0.0))


def elastic_net(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Independent weighted coordinate-descent oracle."""
    mass = float(np.sum(weights))
    coefficient = np.zeros((N_FEATURES + 1, y.shape[1]), dtype=np.float64)
    for output in range(y.shape[1]):
        beta = np.zeros(N_FEATURES, dtype=np.float64)
        intercept = float(np.sum(weights * y[:, output]) / mass)
        for _ in range(5000):
            old_intercept = intercept
            prediction = x @ beta
            intercept = float(np.sum(weights * (y[:, output] - prediction)) / mass)
            residual = y[:, output] - intercept - prediction
            delta = abs(intercept - old_intercept)
            for feature in range(N_FEATURES):
                old_beta = beta[feature]
                residual = residual + x[:, feature] * old_beta
                rho = float(np.sum(weights * x[:, feature] * residual) / mass)
                z = float(np.sum(weights * x[:, feature] * x[:, feature]) / mass)
                beta[feature] = soft_threshold(rho, ALPHA * L1_RATIO) / (
                    z + ALPHA * (1.0 - L1_RATIO)
                )
                residual = residual - x[:, feature] * beta[feature]
                delta = max(delta, abs(beta[feature] - old_beta))
            scale = 1.0 + max(abs(intercept), float(np.max(np.abs(beta))))
            if delta <= 1.0e-12 * scale:
                break
        else:
            raise RuntimeError("NumPy elastic-net oracle did not converge")
        coefficient[:, output] = np.r_[intercept, beta]
    return coefficient


def evaluate_fixture() -> dict[str, np.ndarray]:
    x, y, _, x_dot, theta_dot, u = fixture()
    coefficient = elastic_net(x, y, fixture()[2])
    design = np.column_stack((np.ones(N_SAMPLES), x))
    design_dot = np.column_stack((np.zeros(N_SAMPLES), x_dot))
    coefficient_dot = theta_dot.reshape(coefficient.shape, order="F")
    prediction = design @ coefficient
    prediction_dot = design_dot @ coefficient + design @ coefficient_dot
    theta_bar = (design.T @ u).reshape(-1, order="F")
    x_bar = u @ coefficient[1:, :].T
    plus = np.column_stack((np.ones(N_SAMPLES), x + JVP_STEP * x_dot)) @ (
        coefficient + JVP_STEP * coefficient_dot
    )
    minus = np.column_stack((np.ones(N_SAMPLES), x - JVP_STEP * x_dot)) @ (
        coefficient - JVP_STEP * coefficient_dot
    )
    fd_error = float(np.max(np.abs(prediction_dot - (plus - minus) / (2.0 * JVP_STEP))))
    adjoint_error = float(abs(
        np.sum(u * prediction_dot)
        - (np.sum(theta_bar * theta_dot) + np.sum(x_bar * x_dot))
    ))
    if fd_error > 3.0e-9 or adjoint_error > 3.0e-12:
        raise RuntimeError(
            f"elastic-net derivative oracle failed: finite_difference={fd_error:.3e}, "
            f"adjoint={adjoint_error:.3e}"
        )
    vector_coefficient = elastic_net(x, y[:, :1], fixture()[2])
    return {
        "fit_matrix": coefficient,
        "fit_vector": vector_coefficient,
        "predict_matrix": prediction,
        "predict_vector": (design @ vector_coefficient)[:, 0],
        "predict_jvp": prediction_dot,
        "predict_vjp_theta": theta_bar,
        "predict_vjp_x": x_bar,
    }


def timed(operation: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = None
    for _ in range(REPETITIONS):
        value = operation()
    return value, (time.perf_counter() - started) / REPETITIONS


def oracle_rows(details: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    x, y, weights, x_dot, theta_dot, u = fixture()
    expected = evaluate_fixture()
    design = np.column_stack((np.ones(N_SAMPLES), x))
    design_dot = np.column_stack((np.zeros(N_SAMPLES), x_dot))
    coefficient_dot = theta_dot.reshape((N_FEATURES + 1, N_OUTPUTS), order="F")
    operations: dict[str, Any] = {
        "fit_matrix": lambda: elastic_net(x, y, weights),
        "fit_vector": lambda: elastic_net(x, y[:, :1], weights),
        "predict_matrix": lambda: design @ expected["fit_matrix"],
        "predict_vector": lambda: (design @ expected["fit_vector"])[:, 0],
        "predict_jvp": lambda: design_dot @ expected["fit_matrix"] + design @ coefficient_dot,
        "predict_vjp_theta": lambda: (design.T @ u).reshape(-1, order="F"),
        "predict_vjp_x": lambda: u @ expected["fit_matrix"][1:, :].T,
    }
    rows: list[dict[str, Any]] = []
    for workload, operation in operations.items():
        actual, seconds = timed(operation)
        actual_array = np.asarray(actual)
        target = expected[workload]
        error = float(np.max(np.abs(actual_array - target)))
        if error > 1.0e-13:
            raise RuntimeError(f"NumPy {workload} self-check failed: {error:.3e}")
        rows.append(base(
            details, workload=workload,
            phase="fit" if workload.startswith("fit") else "predict",
            n_outputs=1 if workload in ("fit_vector", "predict_vector") else N_OUTPUTS,
            seconds_per_operation=seconds, metric="l2_norm",
            value=float(np.linalg.norm(actual_array)), max_abs_error=error,
            notes=f"shape={list(actual_array.shape)}; derivative products use fixed fit; "
                  f"independent coordinate descent",
        ))
    return rows, expected


def unavailable_rows(details: dict[str, str], reason: str) -> list[dict[str, Any]]:
    names = ("fit_matrix", "fit_vector", "predict_matrix", "predict_vector",
             "predict_jvp", "predict_vjp_theta", "predict_vjp_x")
    return [base(
        details, workload=name,
        phase="fit" if name.startswith("fit") else "predict",
        backend="fortml", device="cpu", status="unavailable", repetitions="",
        seconds_per_operation="", metric="l2_norm", value="", max_abs_error="",
        oracle="FortML complete-array release-app protocol", notes=reason,
    ) for name in names]


def device_refusal_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    """Record the explicit CUDA boundary without retaining a fake timing."""
    names = ("fit_matrix", "fit_vector", "predict_matrix", "predict_vector",
             "predict_jvp", "predict_vjp_theta", "predict_vjp_x")
    return [base(
        details, workload=name,
        phase="fit" if name.startswith("fit") else "predict",
        backend="fortml", device="cuda", status="unavailable", repetitions="",
        seconds_per_operation="", metric="l2_norm", value="", max_abs_error="",
        oracle="FortML device capability boundary; no CUDA execution",
        notes="elastic-net device_supported(CUDA)=false; no resident CUDA kernel",
    ) for name in names]


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               expected: dict[str, np.ndarray]) -> list[dict[str, Any]]:
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
                         env=environment, capture_output=True, text=True, check=True)
    records: dict[str, dict[int, tuple[float, float]]] = {}
    for line in run.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 5 or fields[0] != "elastic_net":
            continue
        name = fields[1]
        if name not in expected:
            continue
        try:
            index, value, seconds = int(fields[2]), float(fields[3]), float(fields[4])
        except ValueError as error:
            raise RuntimeError(f"invalid FortML elastic-net protocol row: {line!r}") from error
        records.setdefault(name, {})[index] = (value, seconds)
    rows: list[dict[str, Any]] = []
    for name, target_array in expected.items():
        flat_expected = target_array.reshape(-1, order="F")
        record = records.get(name, {})
        if set(record) != set(range(1, flat_expected.size + 1)):
            raise RuntimeError(f"FortML elastic-net protocol omitted complete {name} array")
        actual = np.array([record[index][0] for index in range(1, flat_expected.size + 1)])
        error = float(np.max(np.abs(actual - flat_expected)))
        if error > ORACLE_TOLERANCE:
            raise RuntimeError(f"FortML elastic-net {name} oracle mismatch: {error:.3e}")
        seconds = float(np.median([record[index][1] for index in record]))
        rows.append(base(
            details, workload=name,
            phase="fit" if name.startswith("fit") else "predict",
            backend="fortml", status="pass", seconds_per_operation=seconds,
            metric="l2_norm", value=float(np.linalg.norm(actual)), max_abs_error=error,
            oracle="FortML complete-array release-app protocol",
            notes=f"target={target}; entries={flat_expected.size}",
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/elastic_net.csv"))
    parser.add_argument("--target", default="fortml_bench_elastic_net")
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
    rows.extend(device_refusal_rows(details))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
