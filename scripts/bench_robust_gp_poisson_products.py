#!/usr/bin/env python3
"""Independent oracle and release timing for Poisson GP products.

The likelihood rows use the direct Poisson log-rate formula.  The posterior
row reconstructs the dense Laplace Newton mode independently and evaluates
the fixed-state latent HVP ``-K^{-1}d - exp(f)d``.  Neither oracle reuses
FortML factorization or derivative code.
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
    "workload", "phase", "backend", "device", "status", "n_samples",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
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


def rbf(x_left: np.ndarray, x_right: np.ndarray, variance: float,
        lengthscale: float) -> np.ndarray:
    difference = x_left[:, None] - x_right[None, :]
    return variance * np.exp(-0.5 * difference * difference / lengthscale**2)


def likelihood_oracle() -> tuple[float, float]:
    counts = np.array([0.0, 1.0, 3.0, 2.0])
    log_rate = np.array([-0.3, 0.2, 0.8, -0.1])
    direction = np.array([0.4, -0.2, 0.1, 0.3])
    value = np.sum(counts * log_rate - np.exp(log_rate) -
                   np.array([math.lgamma(y + 1.0) for y in counts]))
    hvp = -np.exp(log_rate) * direction
    return float(value), float(np.sum(hvp))


def posterior_oracle() -> float:
    x = -1.4 + 0.4 * np.arange(8, dtype=np.float64)
    counts = np.maximum(0.0, np.rint(3.0 * np.exp(0.6 * x)))
    covariance = rbf(x, x, 1.0, 0.9) + 1.0e-8 * np.eye(8)
    mode = np.zeros(8)
    for _ in range(200):
        rate = np.exp(mode)
        root = np.sqrt(rate)
        system = np.eye(8) + root[:, None] * covariance * root[None, :]
        b = rate * mode + counts - rate
        solved = np.linalg.solve(system, root * (covariance @ b))
        updated = covariance @ (b - root * solved)
        if np.max(np.abs(updated - mode)) <= 1.0e-10:
            mode = updated
            break
        mode = updated
    direction = np.zeros(8)
    direction[0] = 0.25
    posterior_hvp = -np.linalg.solve(covariance, direction) - np.exp(mode) * direction
    return float(np.sum(posterior_hvp))


def parse_app(stdout: str, details: dict[str, object], rows: list[dict[str, object]]) -> None:
    pattern = re.compile(
        r"^robust_gp_poisson_products,(likelihood|posterior),cpu,seconds,"
        r"\s*([0-9Ee+.-]+),hvp_checksum,\s*([0-9Ee+.-]+)$"
    )
    refusal_pattern = re.compile(
        r"^robust_gp_poisson_products,device,cuda,refused,(\d+)$"
    )
    found: dict[str, tuple[float, float]] = {}
    refusal: list[int] = []
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            found[match.group(1)] = (float(match.group(2)), float(match.group(3)))
        match = refusal_pattern.match(line.strip())
        if match:
            refusal.append(int(match.group(1)))
    if set(found) != {"likelihood", "posterior"}:
        raise RuntimeError(f"release app emitted rows {sorted(found)}")
    expected = {"likelihood": likelihood_oracle()[1], "posterior": posterior_oracle()}
    for phase, (seconds, checksum) in found.items():
        error = abs(checksum - expected[phase])
        if error > 2.0e-10:
            raise RuntimeError(f"{phase} HVP checksum error {error:g}")
        rows.append({**details, "workload": "robust_gp_poisson_products",
                     "phase": f"release_{phase}", "backend": "fortml",
                     "device": "cpu", "status": "pass", "n_samples": 8,
                     "repetitions": 64, "seconds_per_operation": seconds,
                     "metric": "hvp_checksum", "value": checksum,
                     "max_abs_error": error,
                     "oracle": "independent NumPy Poisson Laplace HVP",
                     "notes": "fixed-state log-rate product"})
    if refusal != [3]:
        raise RuntimeError(f"unexpected CUDA refusal codes: {refusal}")
    rows.append({**details, "workload": "robust_gp_poisson_products",
                 "phase": "device_boundary", "backend": "fortml", "device": "cuda",
                 "status": "refused", "n_samples": 8, "repetitions": 0,
                 "metric": "resident_laplace", "value": "nan", "max_abs_error": 0.0,
                 "oracle": "FORTNUM_NOT_IMPLEMENTED",
                 "notes": "no resident CUDA Poisson Laplace solve"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/robust_gp_poisson_products.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    likelihood_value, likelihood_hvp = likelihood_oracle()
    posterior_hvp = posterior_oracle()
    rows: list[dict[str, object]] = []
    for phase, metric, value, notes in (
            ("oracle_likelihood", "log_likelihood", likelihood_value,
             "direct y*log(rate)-rate-log_gamma(y+1)"),
            ("oracle_likelihood_hvp", "hvp_checksum", likelihood_hvp,
             "direct diagonal Poisson Hessian"),
            ("oracle_posterior_hvp", "hvp_checksum", posterior_hvp,
             "independent dense Laplace Newton mode")):
        rows.append({**details, "workload": "robust_gp_poisson_products", "phase": phase,
                     "backend": "numpy", "device": "cpu", "status": "pass",
                     "n_samples": 8, "repetitions": 0, "seconds_per_operation": "",
                     "metric": metric, "value": value, "max_abs_error": 0.0,
                     "oracle": "independent NumPy formula", "notes": notes})
    if args.skip_fortml:
        rows.append({**details, "workload": "robust_gp_poisson_products",
                     "phase": "behavioral_gate", "backend": "fortml", "device": "cpu",
                     "status": "skipped", "n_samples": 8, "repetitions": 0,
                     "metric": "tests_passed", "value": "nan", "max_abs_error": 0.0,
                     "oracle": "test_robust_gp_poisson_products", "notes": "--skip-fortml"})
    else:
        environment = os.environ.copy()
        environment.update({"FO_SCAN_FALLBACK": "regex", "OMP_NUM_THREADS": "1"})
        subprocess.run(["fo", "test", "test_robust_gp_poisson_products"],
                       cwd=fortml, env=environment, check=True)
        rows.append({**details, "workload": "robust_gp_poisson_products",
                     "phase": "behavioral_gate", "backend": "fortml", "device": "cpu",
                     "status": "pass", "n_samples": 8, "repetitions": 0,
                     "metric": "tests_passed", "value": 1.0, "max_abs_error": 0.0,
                     "oracle": "independent Fortran product and refusal test",
                     "notes": "value/gradient/JVP/VJP/HVP and typed CUDA boundary"})
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                       env=environment, check=True, capture_output=True, text=True)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_robust_gp_poisson_products"],
            cwd=fortml, env=environment, check=True, capture_output=True, text=True)
        parse_app(completed.stdout, details, rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
