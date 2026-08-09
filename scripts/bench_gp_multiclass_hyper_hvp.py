#!/usr/bin/env python3
"""Independent oracle and release timing for multiclass GP HVPs.

The NumPy oracle refits each one-vs-rest latent Laplace mode at central
hyperparameter probes and differences the independent envelope gradient.  It
therefore checks the block packing and implicit mode derivative without
reusing FortML's HVP implementation.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_classes", "n_parameters", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
X = np.array([
    [-0.1, 1.9], [0.1, 2.1], [0.2, 1.8], [-0.1, -0.1], [0.1, 0.2],
    [0.3, 0.0], [1.9, 0.0], [2.1, 0.2], [1.8, 0.3],
], dtype=np.float64)
LABELS = np.array([42, 42, 42, -7, -7, -7, 10, 10, 10], dtype=np.int64)
CLASSES = np.array([-7, 10, 42], dtype=np.int64)
THETA = np.log(np.array([1.5, 0.55], dtype=np.float64))
DIRECTION = np.array([0.021, -0.014, 0.017, -0.011, 0.013, -0.009], dtype=np.float64)
JITTER = 1.0e-7


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def fit(theta: np.ndarray, target: int) -> np.ndarray:
    variance, lengthscale = np.exp(theta)
    delta = X[:, None, :] - X[None, :, :]
    squared_distance = np.sum(delta * delta, axis=2)
    signal = variance * np.exp(-0.5 * squared_distance / lengthscale**2)
    covariance = signal + JITTER * np.eye(X.shape[0])
    encoded = np.where(LABELS == target, 1.0, -1.0)
    mode = np.zeros(X.shape[0])
    for _ in range(100):
        eta = encoded * mode
        probability = stable_sigmoid(eta)
        gradient = 1.0 - probability
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-12)
        sqrt_w = np.sqrt(curvature)
        b = curvature * mode + encoded * gradient
        system = np.eye(X.shape[0]) + sqrt_w[:, None] * covariance * sqrt_w[None, :]
        rhs = np.linalg.solve(system, sqrt_w * (covariance @ b))
        mode_new = covariance @ (b - sqrt_w * rhs)
        if np.max(np.abs(mode_new - mode)) / max(1.0, np.max(np.abs(mode))) <= 1.0e-10:
            mode = mode_new
            break
        mode = mode_new
    alpha = np.linalg.solve(covariance, mode)
    d_signal = (
        signal,
        signal * squared_distance / lengthscale**2,
    )
    return np.array([0.5 * alpha @ block @ alpha for block in d_signal])


def oracle() -> tuple[np.ndarray, float]:
    step = 2.0e-4
    result = np.empty_like(DIRECTION)
    for class_index, target in enumerate(CLASSES):
        local = DIRECTION[2 * class_index:2 * class_index + 2]
        plus = fit(THETA + step * local, int(target))
        minus = fit(THETA - step * local, int(target))
        result[2 * class_index:2 * class_index + 2] = (plus - minus) / (2.0 * step)
    return result, float(np.linalg.norm(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_multiclass_hyper_hvp.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    expected, expected_norm = oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_multiclass_hyper_hvp", "backend": "numpy",
                    "device": "cpu", "n_samples": X.shape[0], "n_classes": 3,
                    "n_parameters": DIRECTION.size})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", status="pass", metric="hvp_norm",
        value=expected_norm, max_abs_error=0.0,
        oracle="independent NumPy OVR Laplace refit-gradient central difference",
        notes=f"hvp_sum={np.sum(expected):.16e}")
    if args.skip_fortml:
        add(phase="behavioral_gate", backend="fortml", status="skipped",
            metric="tests_passed", value="nan", oracle="test_gp_multiclass_classification",
            notes="--skip-fortml")
    else:
        environment = os.environ.copy()
        environment.update({"FO_SCAN_FALLBACK": "regex", "OMP_NUM_THREADS": "1"})
        subprocess.run(["fo", "test", "test_gp_multiclass_classification"],
                       cwd=fortml, env=environment, check=True)
        add(phase="behavioral_gate", backend="fortml", status="pass",
            metric="tests_passed", value=1.0, max_abs_error=0.0,
            oracle="independent Fortran refit FD, adjoint, and device tests",
            notes="multiclass gradient/JVP/VJP/HVP and transactional boundaries")
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                       env=environment, check=True, capture_output=True, text=True)
        completed = subprocess.run(
            ["fo", "exec", "--target", "fortml_bench_gp_multiclass_hyper_hvp", "--no-build"],
            cwd=fortml, env=environment, check=True, capture_output=True, text=True)
        match = re.search(
            r"gp_multiclass_hyper_hvp,cpu,seconds,\s*([0-9Ee+.-]+),sum,\s*"
            r"([0-9Ee+.-]+),norm,\s*([0-9Ee+.-]+)", completed.stdout)
        if match is None:
            raise RuntimeError(f"release app emitted unexpected output:\n{completed.stdout}")
        seconds, observed_sum, observed_norm = map(float, match.groups())
        error = max(abs(observed_sum - np.sum(expected)),
                    abs(observed_norm - expected_norm))
        if error > 4.0e-4:
            raise RuntimeError(f"multiclass HVP checksum error {error:g}")
        add(phase="release_app", backend="fortml", status="pass",
            metric="hvp_checksum", value=observed_sum, seconds_per_operation=seconds,
            max_abs_error=error,
            oracle="independent NumPy OVR Laplace refit-gradient central difference",
            notes="three sorted classes; block-packed kernel log parameters")
        refusal = re.search(
            r"gp_multiclass_hyper_hvp,device,cuda,refused,(\d+)", completed.stdout)
        if refusal is None or int(refusal.group(1)) != 3:
            raise RuntimeError(f"unexpected CUDA refusal output:\n{completed.stdout}")
        add(phase="device_boundary", backend="fortml", device="cuda", status="refused",
            metric="resident_multiclass_hvp", value="nan", max_abs_error=0.0,
            oracle="FORTNUM_NOT_IMPLEMENTED",
            notes="resident multiclass Laplace factorization/HVP graph is not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
