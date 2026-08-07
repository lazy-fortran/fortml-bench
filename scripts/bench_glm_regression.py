#!/usr/bin/env python3
"""Correctness-gated weighted Poisson/Gamma GLM benchmark.

The NumPy reference solves the same bounded, L2-regularized log-link
objectives with an independent Newton implementation.  FortML's release app
reports fit timing, prediction means, and objective values.  CUDA remains an
explicit unavailable row until a resident positive-link kernel is linked;
CPU timings are never relabeled as accelerator evidence.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "family", "phase", "backend", "device", "status",
    "n_samples", "n_features", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(line[3:].strip() not in ignored_names for line in status.splitlines())
    return value + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = 256
    i = np.arange(1, n + 1, dtype=np.float64)
    x = np.column_stack((
        -2.0 + 4.0 * (i - 1.0) / (n - 1.0),
        np.sin(0.17 * i),
        np.cos(0.11 * i),
    ))
    eta = 0.25 + 0.55 * x[:, 0] - 0.2 * x[:, 1] + 0.15 * x[:, 2]
    poisson = np.exp(eta)
    gamma = np.exp(eta + 0.08 * np.sin(0.07 * i))
    weights = 0.75 + (i.astype(np.int64) % 5) / 5.0
    return x, poisson, gamma, weights


def objective_newton(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, family: str,
    alpha: float = 0.05, dispersion: float = 1.0,
) -> tuple[np.ndarray, float]:
    design = np.column_stack((np.ones(x.shape[0]), x))
    theta = np.zeros(design.shape[1], dtype=np.float64)
    mass = float(np.sum(weights))
    for _ in range(1000):
        eta = design @ theta
        mu = np.exp(np.minimum(eta, 700.0))
        if family == "poisson":
            residual = mu - y
            value_terms = mu - y * eta
            curvature = mu
        else:
            residual = 1.0 - y / mu
            value_terms = y / mu + eta
            curvature = y / mu
            residual /= dispersion
            value_terms /= dispersion
            curvature /= dispersion
        gradient = (design.T @ (weights * residual)) / mass
        gradient[1:] += alpha * theta[1:]
        hessian = (design.T * (weights * curvature)) @ design / mass
        hessian[1:, 1:] += alpha * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        candidate = np.clip(theta - step, -30.0, 30.0)
        if np.max(np.abs(candidate - theta)) < 1.0e-12:
            theta = candidate
            break
        theta = candidate
    eta = design @ theta
    mu = np.exp(eta)
    if family == "poisson":
        terms = mu - y * eta
    else:
        terms = (y / mu + eta) / dispersion
    value = float(np.sum(weights * terms) / mass + 0.5 * alpha * np.sum(theta[1:] ** 2))
    return np.exp(eta), value


def run_app(fortml: Path, target: str) -> list[str]:
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/glm_regression.csv"))
    parser.add_argument("--target", default="fortml_bench_glm_regression")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    lines = run_app(fortml, args.target)
    x, poisson, gamma, weights = fixture()
    oracle = {
        "poisson": objective_newton(x, poisson, weights, "poisson"),
        "gamma": objective_newton(x, gamma, weights, "gamma"),
    }
    # Ignore documentation-only and unrelated concurrent slices when pinning
    # this numerical lane; source/app/test changes remain visible as dirty.
    fortml_rev = revision(fortml, (
        fortml / "verification" / "fortml-gfortran.txt",
        fortml / "README.md", fortml / "ROADMAP.md", fortml / "docs" / "API.md",
        fortml / "app" / "fortml_bench_kernel_catalog.f90",
    ))
    bench_rev = revision(root, (
        root / "results" / "glm_regression.csv",
        root / "results" / "radius_neighbors.csv",
    ))
    rows: list[dict[str, str]] = []

    def row(**kwargs: object) -> dict[str, str]:
        result = {field: "" for field in FIELDS}
        result.update({
            "workload": "glm_regression", "backend": "fortml", "device": "cpu",
            "status": "pass", "compiler": "gfortran", "flags": "-O3",
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "fortml_revision": fortml_rev, "benchmark_revision": bench_rev,
            "oracle": "numpy_newton_log_link", "n_samples": str(x.shape[0]),
            "n_features": str(x.shape[1]),
        })
        result.update({key: str(value) for key, value in kwargs.items()})
        return result

    seen: set[str] = set()
    for line in lines:
        fields = line.split(",")
        if not fields or fields[0] != "glm_family":
            continue
        if len(fields) != 10:
            raise RuntimeError(f"unexpected GLM release row: {line}")
        family_code = int(fields[1])
        family = "poisson" if family_code == 1 else "gamma" if family_code == 2 else "unknown"
        if family not in oracle:
            raise RuntimeError(f"unknown GLM family code: {family_code}")
        # The app row is family,n_samples,fit_seconds,mean_prediction,objective.
        n_samples = fields[3]
        seconds = float(fields[5])
        mean_prediction = float(fields[7])
        app_value = float(fields[9])
        expected_prediction, expected_value = oracle[family]
        error = max(abs(app_value - expected_value), abs(
            mean_prediction - float(np.mean(expected_prediction))))
        if not np.isfinite(error) or error > 2.0e-6:
            raise RuntimeError(f"{family} NumPy oracle mismatch: {error:g}")
        seen.add(family)
        rows.append(row(family=family, phase="fit_predict", n_samples=n_samples,
                        seconds_per_operation=seconds, metric="objective",
                        value=app_value, max_abs_error=error,
                        notes=f"mean_prediction={mean_prediction:.16g}"))
    if seen != set(oracle):
        raise RuntimeError(f"release app omitted GLM families: {set(oracle) - seen}")
    rows.append(row(family="all", phase="predict", device="cuda", status="unavailable",
                    seconds_per_operation="0.0", metric="objective", value="nan",
                    max_abs_error="nan", oracle="typed_device_contract",
                    notes="no resident GLM CUDA kernel; FORTNUM_NOT_IMPLEMENTED"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
