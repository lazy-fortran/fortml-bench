#!/usr/bin/env python3
"""Independent oracle and release timing for binary Laplace-GP HVPs.

The NumPy implementation refits each perturbed kernel and central-differences
the envelope gradient.  This checks the implicit-mode Fortran product rather
than replaying its posterior-factorization algebra.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "likelihood",
    "n_samples", "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
X = np.linspace(-1.5, 1.5, 24, dtype=np.float64)
LABELS = np.where(X >= 0.0, 11, -7).astype(np.int64)
JITTER = 1.0e-7
THETA = np.log(np.array([1.35, 0.79], dtype=np.float64))
DIRECTION = np.array([0.07, -0.04], dtype=np.float64)


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


def stable_sigmoid(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def likelihood_terms(eta: np.ndarray, likelihood: str) -> tuple[np.ndarray, np.ndarray]:
    if likelihood == "logistic":
        probability = stable_sigmoid(eta)
        return 1.0 - probability, np.maximum(probability * (1.0 - probability), 1.0e-12)
    probability = 0.5 * np.vectorize(math.erfc)(-eta / np.sqrt(2.0))
    density = np.exp(-0.5 * eta * eta) / np.sqrt(2.0 * np.pi)
    ratio = density / np.maximum(probability, 1.0e-14)
    ratio = np.where(probability > 1.0e-14, ratio,
                     np.maximum(1.0, -eta) + 1.0 / np.maximum(1.0, -eta))
    return ratio, np.maximum(ratio * (ratio + eta), 1.0e-12)


def fit(theta: np.ndarray, likelihood: str) -> tuple[np.ndarray, np.ndarray]:
    variance, lengthscale = np.exp(theta)
    distance = X[:, None] - X[None, :]
    signal = variance * np.exp(-0.5 * distance * distance / lengthscale**2)
    covariance = signal + JITTER * np.eye(X.size)
    encoded = np.where(LABELS == np.max(LABELS), 1.0, -1.0)
    mode = np.zeros(X.size)
    for _ in range(100):
        eta = encoded * mode
        gradient, curvature = likelihood_terms(eta, likelihood)
        sqrt_w = np.sqrt(curvature)
        b = curvature * mode + encoded * gradient
        system = np.eye(X.size) + sqrt_w[:, None] * covariance * sqrt_w[None, :]
        rhs = np.linalg.solve(system, sqrt_w * (covariance @ b))
        mode_new = covariance @ (b - sqrt_w * rhs)
        step = np.max(np.abs(mode_new - mode)) / max(1.0, np.max(np.abs(mode)))
        mode += mode_new - mode
        if step <= 1.0e-10:
            break
    alpha = np.linalg.solve(covariance, mode)
    d_signal = (
        signal,
        signal * distance * distance / lengthscale**2,
    )
    gradient = np.array([0.5 * alpha @ block @ alpha for block in d_signal])
    return alpha, gradient


def oracle(likelihood: str) -> tuple[np.ndarray, np.ndarray]:
    _, gradient_plus = fit(THETA + 2.0e-4 * DIRECTION, likelihood)
    _, gradient_minus = fit(THETA - 2.0e-4 * DIRECTION, likelihood)
    hvp = (gradient_plus - gradient_minus) / (4.0e-4)
    return hvp, gradient_plus - gradient_minus


def row(details: dict[str, object], **values: object) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({"device": "cpu", "n_samples": X.size, "repetitions": 32})
    result.update(values)
    return result


def parse_app(stdout: str, details: dict[str, object], rows: list[dict[str, object]]) -> None:
    pattern = re.compile(
        r"^gp_classification_hvp,(logistic|probit\s*),cpu,seconds,"
        r"\s*([0-9Ee+.-]+),sum,\s*([0-9Ee+.-]+),norm,\s*([0-9Ee+.-]+)$"
    )
    refusal_pattern = re.compile(
        r"^gp_classification_hvp,device,cuda,refused,(\d+)$"
    )
    found: dict[str, tuple[float, float, float]] = {}
    refusal: list[int] = []
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            found[match.group(1).strip()] = tuple(float(match.group(i)) for i in (2, 3, 4))
        cuda = refusal_pattern.match(line.strip())
        if cuda:
            refusal.append(int(cuda.group(1)))
    if set(found) != {"logistic", "probit"}:
        raise RuntimeError(f"release app emitted likelihoods {sorted(found)}")
    for likelihood, (seconds, observed_sum, observed_norm) in found.items():
        expected, _ = oracle(likelihood)
        error = max(abs(observed_sum - np.sum(expected)),
                    abs(observed_norm - np.linalg.norm(expected)))
        if error > 4.0e-4:
            raise RuntimeError(f"{likelihood} release checksum error {error:g}")
        rows.append(row(details, workload="gp_classification_hvp", phase="release_app",
                        backend="fortml", likelihood=likelihood, status="pass",
                        seconds_per_operation=seconds, metric="hvp_checksum",
                        value=observed_sum, max_abs_error=error,
                        oracle="independent NumPy refit-gradient central difference",
                        notes="implicit converged Laplace mode; RBF log parameters"))
    if refusal != [3]:
        raise RuntimeError(f"unexpected CUDA refusal codes: {refusal}")
    rows.append(row(details, workload="gp_classification_hvp", phase="device_boundary",
                    backend="fortml", device="cuda", status="refused",
                    likelihood="both", metric="resident_hvp", value="nan",
                    max_abs_error=0.0, oracle="FORTNUM_NOT_IMPLEMENTED",
                    notes="no resident binary Laplace covariance/HVP kernel"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_classification_hvp.csv"))
    parser.add_argument("--target", default="fortml_bench_gp_classification_hvp")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = []
    for likelihood in ("logistic", "probit"):
        expected, _ = oracle(likelihood)
        rows.append(row(details, workload="gp_classification_hvp", phase="oracle",
                        backend="numpy", likelihood=likelihood, status="pass",
                        metric="hvp_checksum", value=float(np.sum(expected)),
                        max_abs_error=0.0,
                        oracle="independent NumPy Laplace refit and envelope-gradient FD",
                        notes=f"hvp_norm={np.linalg.norm(expected):.16e}"))
    if args.skip_fortml:
        rows.append(row(details, workload="gp_classification_hvp", phase="behavioral_gate",
                        backend="fortml", status="skipped", metric="tests_passed",
                        value="nan", oracle="test_gp_classification_hvp",
                        notes="--skip-fortml"))
    else:
        environment = os.environ.copy()
        environment.update({"FO_SCAN_FALLBACK": "regex", "OMP_NUM_THREADS": "1"})
        subprocess.run(["fo", "test", "test_gp_classification_hvp"], cwd=fortml,
                       env=environment, check=True)
        rows.append(row(details, workload="gp_classification_hvp", phase="behavioral_gate",
                        backend="fortml", status="pass", metric="tests_passed", value=1.0,
                        max_abs_error=0.0, oracle="independent Fortran refit FD and device oracle",
                        notes="logistic/probit implicit HVP; transactional and CUDA boundaries"))
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                       env=environment, check=True, capture_output=True, text=True)
        completed = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                                   env=environment, check=True, capture_output=True, text=True)
        parse_app(completed.stdout, details, rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
