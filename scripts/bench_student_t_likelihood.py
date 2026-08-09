#!/usr/bin/env python3
"""Benchmark the fixed-state Student-t GP likelihood coordinate.

The NumPy path rebuilds the RBF covariance, Cholesky factor, and Student-t
marginal density independently. Central differences in theta=log(nu-2) are
the behavioral oracle for the released JVP and HVP values.
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


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.5, 1.5, 7, dtype=np.float64)[:, None]
    y = np.sin(1.7 * x[:, 0]) + 0.25 * x[:, 0]
    return x, y


def lml(theta: float) -> float:
    x, y = fixture()
    difference = x[:, None, :] - x[None, :, :]
    gram = np.exp(-0.5 * np.sum(difference * difference, axis=2) / 0.8**2)
    gram += (0.01 + 1.0e-10) * np.eye(x.shape[0])
    factor = np.linalg.cholesky(gram)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
    beta = float(y @ alpha)
    nu = 2.0 + math.exp(theta)
    logdet = 2.0 * float(np.sum(np.log(np.diag(factor))))
    n = float(x.shape[0])
    return (
        math.lgamma(0.5 * (nu + n)) - math.lgamma(0.5 * nu)
        - 0.5 * n * math.log((nu - 2.0) * math.pi) - 0.5 * logdet
        - 0.5 * (nu + n) * math.log(1.0 + beta / (nu - 2.0))
    )


def parse_probe(stdout: str) -> dict[str, float | int]:
    records: dict[str, float | int] = {}
    for line in stdout.splitlines():
        if not line.startswith("student_t_likelihood_"):
            continue
        key, value = line.strip().split(",", maxsplit=1)
        records[key] = int(value) if key.startswith("student_t_likelihood_cuda_") else float(value)
    required = {
        "student_t_likelihood_theta", "student_t_likelihood_value",
        "student_t_likelihood_jvp", "student_t_likelihood_vjp",
        "student_t_likelihood_hvp", "student_t_likelihood_cuda_jvp",
        "student_t_likelihood_cuda_vjp", "student_t_likelihood_cuda_hvp",
    }
    missing = required - records.keys()
    if missing:
        raise RuntimeError(f"Student-t likelihood release probe omitted {sorted(missing)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/student_t_likelihood.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    environment = os.environ.copy()
    environment["FO_SCAN_FALLBACK"] = "regex"
    subprocess.run(["fo", "test", "test_student_t_process"], cwd=fortml,
                   env=environment, check=True)
    started = time.perf_counter()
    probe = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_student_t_likelihood"],
                           cwd=fortml, env=environment, check=True, text=True,
                           capture_output=True)
    elapsed = time.perf_counter() - started
    record = parse_probe(probe.stdout)
    theta = float(record["student_t_likelihood_theta"])
    h = 3.0e-4
    reference_value = lml(theta)
    reference_jvp = (lml(theta + h) - lml(theta - h)) / (2.0 * h)
    reference_hvp = (lml(theta + h) - 2.0 * reference_value + lml(theta - h)) / h**2
    value_error = abs(float(record["student_t_likelihood_value"]) - reference_value)
    jvp_error = abs(float(record["student_t_likelihood_jvp"]) - reference_jvp)
    vjp_error = abs(float(record["student_t_likelihood_vjp"]) - reference_jvp)
    hvp_error = abs(float(record["student_t_likelihood_hvp"]) - reference_hvp)
    if value_error > 2.0e-12 or jvp_error > 3.0e-8 or vjp_error > 3.0e-8 or hvp_error > 3.0e-6:
        raise RuntimeError(
            "Student-t likelihood oracle failed: "
            f"value={value_error:.3e}, jvp={jvp_error:.3e}, "
            f"vjp={vjp_error:.3e}, hvp={hvp_error:.3e}"
        )
    cuda_codes = [
        int(record["student_t_likelihood_cuda_jvp"]),
        int(record["student_t_likelihood_cuda_vjp"]),
        int(record["student_t_likelihood_cuda_hvp"]),
    ]
    if cuda_codes != [3, 3, 3]:
        raise RuntimeError(f"Student-t likelihood CUDA refusal codes changed: {cuda_codes}")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "student_t_likelihood", "backend": "fortml",
                    "device": "cpu", "n_samples": 7, "n_queries": 0,
                    "degrees_of_freedom": 4.7})
        row.update(values)
        rows.append(row)

    maximum_error = max(value_error, jvp_error, vjp_error, hvp_error)
    add(phase="fixed_state_products", status="pass", seconds_per_operation=elapsed,
        metric="lml_jvp_vjp_hvp_max_abs", value=maximum_error,
        max_abs_error=maximum_error,
        oracle="independent NumPy Cholesky Student-t density central differences",
        notes=(f"value={value_error:.3e}; jvp={jvp_error:.3e}; "
               f"vjp={vjp_error:.3e}; hvp={hvp_error:.3e}; theta={theta:.6f}"))
    add(phase="public_contract_gate", status="pass", metric="fortml_student_t_test",
        value=1.0, max_abs_error=maximum_error,
        oracle="FortML test_student_t_process behavioral gate",
        notes="finite-difference JVP/HVP, adjoint, update, and device contracts")
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_student_t_factorization", value="nan", max_abs_error="nan",
        oracle="typed FORTNUM_NOT_IMPLEMENTED refusal",
        notes="JVP/VJP/HVP status_code=3; no host fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
