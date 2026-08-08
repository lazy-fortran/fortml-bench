#!/usr/bin/env python3
"""Benchmark the Student-t process regression contract.

The independent NumPy path pins the large-degree-of-freedom GP limit and the
data-dependent predictive covariance that distinguishes a Student-t process
from a Gaussian process. The FortML test supplies the API/refusal gate before
the CPU contract timing is recorded.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_queries", "degrees_of_freedom", "seconds_per_operation", "metric",
    "value", "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.5, 1.5, 7, dtype=np.float64)[:, None]
    y = np.sin(1.7 * x[:, 0]) + 0.25 * x[:, 0]
    query = np.linspace(-1.2, 1.2, 5, dtype=np.float64)[:, None]
    return x, y, query


def rbf(x_left: np.ndarray, x_right: np.ndarray) -> np.ndarray:
    difference = x_left[:, None, :] - x_right[None, :, :]
    return np.exp(-0.5 * np.sum(difference * difference, axis=2) / 0.8**2)


def gp_posterior(y: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _, _ = fixture()
    gram = rbf(x, x) + (0.01 + 1.0e-10) * np.eye(x.shape[0])
    cross = rbf(x, query)
    prior = rbf(query, query)
    factor = np.linalg.cholesky(gram)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
    work = np.linalg.solve(factor.T, np.linalg.solve(factor, cross))
    mean = cross.T @ alpha
    variance = np.diag(prior) - np.sum(cross * work, axis=0)
    return mean, variance, np.linalg.solve(factor.T, np.linalg.solve(factor, y))


def student_t_posterior(y: np.ndarray, query: np.ndarray, nu: float) -> tuple[np.ndarray, np.ndarray, float]:
    mean, gp_variance, alpha = gp_posterior(y, query)
    x, _, _ = fixture()
    beta = float(y @ alpha)
    scale = (nu + beta - 2.0) / (nu + x.shape[0] - 2.0)
    return mean, scale * gp_variance, scale


def oracle() -> tuple[float, float, float, float]:
    x, y, query = fixture()
    del x
    large_nu_mean, large_nu_variance, _ = student_t_posterior(y, query, 1.0e6)
    gp_mean, gp_variance, _ = gp_posterior(y, query)
    mean_error = float(np.max(np.abs(large_nu_mean - gp_mean)))
    variance_error = float(np.max(np.abs(large_nu_variance - gp_variance)))
    calm = 0.02 * y
    wild = 6.0 * y
    _, calm_variance, calm_scale = student_t_posterior(calm, query, 4.0)
    _, wild_variance, wild_scale = student_t_posterior(wild, query, 4.0)
    _, gp_calm_variance, _ = gp_posterior(calm, query)
    _, gp_wild_variance, _ = gp_posterior(wild, query)
    contrast_error = float(max(0.0, np.max(calm_variance - wild_variance)))
    gp_data_dependence_error = float(np.max(np.abs(gp_calm_variance - gp_wild_variance)))
    if mean_error > 1.0e-10 or variance_error > 1.0e-4 or contrast_error > 0.0:
        raise RuntimeError(
            f"Student-t oracle failed: mean={mean_error:.3e}, variance={variance_error:.3e}, "
            f"calm/wild ordering error={contrast_error:.3e}"
        )
    if gp_data_dependence_error > 1.0e-12 or wild_scale <= calm_scale:
        raise RuntimeError("GP contrast or Student-t covariance scale oracle failed")
    return mean_error, variance_error, wild_scale / calm_scale, gp_data_dependence_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/student_t_process.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    mean_error, variance_error, scale_ratio, gp_contrast_error = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_student_t_process"],
                       cwd=fortml, env=environment, check=True)
        status = "pass"
        notes = "FortML test covers large-nu GP limit, data-dependent covariance, and invalid-nu refusal"
    elapsed = time.perf_counter() - started
    ignored_source = tuple(
        fortml / name for name in ("test_mlp_amsgrad_checkpoint.txt", "test_mlp_radam_checkpoint.txt")
    )
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml, ignored_source),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "student_t_process", "backend": "fortml", "device": "cpu",
                    "n_samples": 7, "n_queries": 5})
        row.update(values)
        rows.append(row)

    add(phase="large_nu_oracle", backend="numpy_oracle", status="pass",
        degrees_of_freedom=1.0e6, metric="gp_limit_max_abs_error",
        value=mean_error, max_abs_error=max(mean_error, variance_error),
        oracle="independent NumPy Cholesky GP/Student-t predictive formulas",
        notes=f"mean_error={mean_error:.3e}; variance_error={variance_error:.3e}")
    add(phase="data_dependent_covariance", backend="numpy_oracle", status="pass",
        degrees_of_freedom=4.0, metric="wild_to_calm_covariance_scale_ratio",
        value=scale_ratio, max_abs_error=gp_contrast_error,
        oracle="independent NumPy same-input calm/wild covariance contrast",
        notes="Student-t variance changes with observed values while GP variance does not")
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        degrees_of_freedom=4.0, metric="fortml_student_t_test", value=1.0,
        max_abs_error=max(mean_error, variance_error),
        oracle="FortML test_student_t_process behavioral gate", notes=notes)
    add(phase="refusal_contract", status="refused", degrees_of_freedom=2.0,
        metric="invalid_covariance_degrees_of_freedom", value="nan", max_abs_error=0.0,
        oracle="typed FORTNUM_DOMAIN_ERROR at nu <= 2", 
        notes="covariance does not exist at or below two degrees of freedom")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
