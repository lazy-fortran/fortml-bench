#!/usr/bin/env python3
"""Correctness-gated GP classifier prediction-through-fit JVP benchmark."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


X = np.array([-1.7, -1.15, -0.62, -0.18, 0.14, 0.55, 1.08, 1.63])
LABEL = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
WEIGHT = np.array([0.45, 1.4, 0.0, 0.8, 1.7, 0.6, 1.25, 0.9])
QUERY = np.array([-0.85, 0.05, 0.92])
THETA = np.log(np.array([1.35, 0.72]))
DIRECTION = np.array([0.19, -0.14])
JITTER = 1.0e-7
FD_STEP = 2.0e-5
REPETITIONS = 32
MIN_CURVATURE = 1.0e-12
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
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    for line in status.splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def rbf(theta: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    variance, lengthscale = np.exp(theta)
    delta = left[:, None] - right[None, :]
    return variance * np.exp(-0.5 * (delta / lengthscale) ** 2)


def likelihood_terms(eta: float, likelihood: str) -> tuple[float, float]:
    if likelihood == "logistic":
        if eta >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-eta))
        else:
            exponential = math.exp(eta)
            probability = exponential / (1.0 + exponential)
        gradient = 1.0 - probability
        curvature = max(probability * (1.0 - probability), MIN_CURVATURE)
        return gradient, curvature
    probability = 0.5 * math.erfc(-eta / math.sqrt(2.0))
    density = math.exp(-0.5 * eta * eta) / math.sqrt(2.0 * math.pi)
    if probability > 1.0e-14:
        ratio = density / probability
    else:
        scale = max(1.0, -eta)
        ratio = scale + 1.0 / scale
    return ratio, max(ratio * (ratio + eta), MIN_CURVATURE)


def fit_predict(theta: np.ndarray, likelihood: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = rbf(theta, X, X)
    covariance[np.diag_indices_from(covariance)] += JITTER
    mode = np.zeros(X.size)
    for _ in range(120):
        gradient = np.empty(X.size)
        curvature = np.empty(X.size)
        for index, eta in enumerate(LABEL * mode):
            gradient[index], curvature[index] = likelihood_terms(float(eta), likelihood)
        sqrt_w = np.where(
            WEIGHT > 0.0, np.sqrt(np.maximum(WEIGHT * curvature, MIN_CURVATURE)), 0.0
        )
        b = WEIGHT * curvature * mode + LABEL * WEIGHT * gradient
        system = np.eye(X.size) + sqrt_w[:, None] * covariance * sqrt_w[None, :]
        rhs = sqrt_w * (covariance @ b)
        solution = np.linalg.solve(system, rhs)
        mode_new = covariance @ (b - sqrt_w * solution)
        step_norm = np.max(np.abs(mode_new - mode)) / max(1.0, np.max(np.abs(mode)))
        mode = mode_new
        if step_norm <= 1.0e-11:
            break
    else:
        raise RuntimeError(f"NumPy {likelihood} Laplace iteration did not converge")

    curvature = np.array([
        likelihood_terms(float(eta), likelihood)[1] for eta in LABEL * mode
    ])
    sqrt_w = np.where(
        WEIGHT > 0.0, np.sqrt(np.maximum(WEIGHT * curvature, MIN_CURVATURE)), 0.0
    )
    system = np.eye(X.size) + sqrt_w[:, None] * covariance * sqrt_w[None, :]
    alpha = np.linalg.solve(covariance, mode)
    cross = rbf(theta, X, QUERY)
    mean = cross.T @ alpha
    work = np.linalg.solve(system, sqrt_w[:, None] * cross)
    prior = np.diag(rbf(theta, QUERY, QUERY))
    variance = prior - np.sum(work * work, axis=0)
    if likelihood == "logistic":
        scale = np.sqrt(1.0 + math.pi * variance / 8.0)
        probability = 1.0 / (1.0 + np.exp(-mean / scale))
    else:
        scale = np.sqrt(1.0 + variance)
        probability = np.array([
            0.5 * math.erfc(-float(value) / math.sqrt(2.0))
            for value in mean / scale
        ])
    return mean, variance, probability


def oracle(likelihood: str) -> dict[str, np.ndarray]:
    plus = fit_predict(THETA + FD_STEP * DIRECTION, likelihood)
    minus = fit_predict(THETA - FD_STEP * DIRECTION, likelihood)
    tangent = tuple((p - m) / (2.0 * FD_STEP) for p, m in zip(plus, minus))
    return {"mean_dot": tangent[0], "variance_dot": tangent[1],
            "probability_dot": tangent[2]}


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "device": "cpu", "n_train": X.size, "n_validation": QUERY.size,
        "n_parameters": THETA.size, "evaluations": 1, "repetitions": REPETITIONS,
    })
    row.update(values)
    return row


def parse_app(stdout: str) -> dict[tuple[int, str, int], float]:
    values: dict[tuple[int, str, int], float] = {}
    for line in stdout.splitlines():
        if not line.startswith("gp_classification_implicit,"):
            continue
        fields = [field.strip() for field in line.split(",")]
        likelihood = int(fields[1])
        quantity = fields[2]
        if quantity in {"seconds", "cuda_status"}:
            values[(likelihood, quantity, 0)] = float(fields[3])
        else:
            values[(likelihood, quantity, int(fields[3]))] = float(fields[4])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/gp_classification_implicit_prediction.csv"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    expected = {name: oracle(name) for name in ("logistic", "probit")}

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
        ["fo", "exec", "--no-build", "fortml_bench_gp_classification_implicit_prediction"],
        cwd=fortml, env=environment, capture_output=True, text=True,
    )
    if run.returncode != 0:
        raise RuntimeError(f"FortML implicit prediction app failed:\n{run.stderr[-2000:]}")
    actual = parse_app(run.stdout)

    rows: list[dict[str, Any]] = []
    oracle_name = "independent NumPy Laplace refits and central parameter difference"
    for likelihood_index, likelihood in enumerate(("logistic", "probit"), start=1):
        for quantity in ("mean_dot", "variance_dot", "probability_dot"):
            for index, value in enumerate(expected[likelihood][quantity], start=1):
                rows.append(base(
                    details,
                    workload="gp_classification_implicit_prediction",
                    phase="oracle_jvp",
                    variant=likelihood,
                    backend="numpy_oracle",
                    status="pass",
                    metric=f"{quantity}_{index}",
                    value=float(value),
                    max_abs_error=0.0,
                    oracle=oracle_name,
                    notes=f"direction={DIRECTION.tolist()}; h={FD_STEP:g}",
                ))
                key = (likelihood_index, quantity, index)
                if key not in actual:
                    raise RuntimeError(f"FortML app omitted {key}")
                error = abs(actual[key] - value)
                if error > 4.0e-6:
                    raise RuntimeError(
                        f"FortML {likelihood} {quantity}[{index}] mismatch: {error:.3e}"
                    )
                rows.append(base(
                    details,
                    workload="gp_classification_implicit_prediction",
                    phase="implicit_fit_jvp",
                    variant=likelihood,
                    backend="fortml",
                    status="pass",
                    seconds_per_operation=actual[(likelihood_index, "seconds", 0)],
                    metric=f"{quantity}_{index}",
                    value=actual[key],
                    max_abs_error=error,
                    oracle=oracle_name,
                    notes="weighted fit; packed=[log_variance,log_lengthscale]",
                ))
        cuda_code = int(actual[(likelihood_index, "cuda_status", 0)])
        if cuda_code <= 0:
            raise RuntimeError(f"FortML {likelihood} CUDA request was not refused")
        rows.append(base(
            details,
            workload="gp_classification_implicit_prediction",
            phase="device_contract",
            variant=likelihood,
            backend="fortml",
            device="cuda",
            status="refused",
            metric="resident_implicit_fit_jvp",
            oracle="FortML typed device contract",
            notes=f"no host fallback; FORTNUM_NOT_IMPLEMENTED status={cuda_code}",
        ))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
