#!/usr/bin/env python3
"""Independent dense oracle for exact-GP seeded hyperparameter multistart."""

from __future__ import annotations

import argparse
import csv
import platform
import re
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "dimensions",
    "starts", "successful_starts", "best_start", "evaluations",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)
X = np.linspace(-1.0, 1.0, 10, dtype=np.float64)[:, None]
Y = (np.sin(2.0 * X[:, 0]) + 0.15 * np.cos(3.0 * X[:, 0]))[:, None]
JITTER = 1.0e-10


def real_token(token: str) -> float:
    token = token.strip()
    if "e" not in token.lower():
        token = re.sub(r"(?<=\d)([+-]\d{3})$", r"E\1", token)
    return float(token)


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


def exact_negative_log_likelihood(parameters: np.ndarray) -> float:
    variance, lengthscale, noise = np.exp(parameters)
    distance = (X[:, None, :] - X[None, :, :]) ** 2
    covariance = variance * np.exp(-distance[:, :, 0] / (2.0 * lengthscale**2))
    covariance[np.diag_indices_from(covariance)] += noise + JITTER
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise RuntimeError("dense oracle covariance is not positive definite")
    alpha = np.linalg.solve(covariance, Y[:, 0])
    log_likelihood = (
        -0.5 * Y[:, 0] @ alpha
        - 0.5 * logdet
        - 0.5 * X.shape[0] * np.log(2.0 * np.pi)
    )
    return float(-log_likelihood)


def finite_difference_gradient(parameters: np.ndarray) -> np.ndarray:
    epsilon = 2.0e-5
    gradient = np.empty_like(parameters)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (
            exact_negative_log_likelihood(plus)
            - exact_negative_log_likelihood(minus)
        ) / (2.0 * epsilon)
    return gradient


def make_row(details: dict[str, str], **values: object) -> dict[str, object]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"device": "cpu", "dimensions": X.shape[0]})
    row.update(values)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_hyperparameter_training.csv"))
    parser.add_argument("--target", default="fortml_bench_gp_hyperparameter_training")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    records = {}
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if fields and fields[0].startswith("gp_exact_"):
            records[fields[0]] = fields
    if "gp_exact_multistart" not in records or "gp_exact_cuda_refusal" not in records:
        raise RuntimeError(f"release app rows missing: {sorted(records)}")

    fields = records["gp_exact_multistart"]
    starts = int(fields[1])
    successful = int(fields[2])
    best_start = int(fields[3])
    evaluations = int(fields[4])
    seconds = real_token(fields[5])
    value = real_token(fields[6])
    gradient_norm = real_token(fields[7])
    parameters = np.array([real_token(fields[index]) for index in (8, 9, 10)])
    oracle_value = exact_negative_log_likelihood(parameters)
    oracle_gradient = finite_difference_gradient(parameters)
    error = max(abs(value - oracle_value), abs(gradient_norm - np.linalg.norm(oracle_gradient)))
    if starts != 4 or successful < 1 or not 1 <= best_start <= starts or evaluations < starts:
        raise RuntimeError(
            f"multistart accounting mismatch: starts={starts}, successful={successful}, "
            f"best={best_start}, evaluations={evaluations}"
        )
    if error > 2.0e-4:
        raise RuntimeError(f"exact-GP dense oracle mismatch: {error:.3e}")
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran",
        "flags": "-O3",
        "oracle": "independent NumPy dense Cholesky-equivalent solve and central FD gradient",
    }
    rows = [make_row(
        details, workload="gp_exact_hyperparameter_multistart", phase="train",
        backend="fortml", status="pass", starts=starts,
        successful_starts=successful, best_start=best_start, evaluations=evaluations,
        seconds_per_operation=seconds, metric="negative_log_marginal_likelihood",
        value=value, max_abs_error=error,
        notes="seed=20260807; include_current=true; retained best finite converged state",
    )]
    refusal_code = int(records["gp_exact_cuda_refusal"][1])
    if refusal_code != 3:
        raise RuntimeError(f"CUDA refusal code mismatch: {refusal_code}")
    rows.append(make_row(
        details, workload="gp_exact_hyperparameter_multistart", phase="train",
        backend="fortml", device="cuda", status="unavailable", starts=starts,
        successful_starts=0, best_start=0, evaluations=0,
        seconds_per_operation=0.0, metric="negative_log_marginal_likelihood",
        value="nan", max_abs_error="nan", oracle="typed device contract",
        notes="FORTNUM_NOT_IMPLEMENTED; exact GP factorization is not resident CUDA",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
