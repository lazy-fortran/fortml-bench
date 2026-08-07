#!/usr/bin/env python3
"""Independent oracle for bounded Laplace-GP classification training."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "classes", "repetitions", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)
X = np.array([-1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5], dtype=np.float64)
BINARY = np.array([-7, -7, -7, -7, 11, 11, 11, 11], dtype=np.int64)
MULTI = np.array([42, 42, 42, -7, -7, 11, 11, 11], dtype=np.int64)
JITTER = 1.0e-7


def real_token(token: str) -> float:
    """Parse gfortran ES output even when a narrow field omits ``E``."""
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


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def mode_objective(log_parameters: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, int]:
    variance, lengthscale = np.exp(log_parameters)
    distances = (X[:, None] - X[None, :]) ** 2
    signal = variance * np.exp(-distances / (2.0 * lengthscale**2))
    covariance = signal.copy()
    covariance[np.diag_indices_from(covariance)] += JITTER
    encoded = np.where(labels == np.max(labels), 1.0, -1.0)
    mode = np.zeros(X.size)
    iterations = 0
    for iterations in range(1, 101):
        eta = encoded * mode
        probability = 1.0 / (1.0 + np.exp(-eta))
        likelihood_gradient = 1.0 - probability
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-12)
        sqrt_w = np.sqrt(curvature)
        b = curvature * mode + encoded * likelihood_gradient
        system = np.eye(X.size) + sqrt_w[:, None] * covariance * sqrt_w[None, :]
        rhs = np.linalg.solve(system, sqrt_w * (covariance @ b))
        mode_new = covariance @ (b - sqrt_w * rhs)
        step = np.max(np.abs(mode_new - mode)) / max(1.0, np.max(np.abs(mode)))
        mode += mode_new - mode
        if step <= 1.0e-9:
            break
    alpha = np.linalg.solve(covariance, mode)
    objective = 0.5 * mode @ alpha - np.log(1.0 / (1.0 + np.exp(-encoded * mode))).sum()
    gradients = np.array([
        0.5 * alpha @ (signal @ alpha),
        0.5 * alpha @ ((signal * distances / lengthscale**2) @ alpha),
    ])
    return float(objective), gradients, iterations


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({"device": "cpu", "n_samples": X.size, "n_features": 1})
    result.update(values)
    return result


def parse_app(stdout: str, details: dict[str, str]) -> list[dict[str, object]]:
    records: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if fields and fields[0].startswith("gp_classification_"):
            records[fields[0]] = fields
    binary_parameters = np.array([real_token(records["gp_classification_binary"][8]),
                                  real_token(records["gp_classification_binary"][9])])
    expected_binary = mode_objective(binary_parameters, BINARY)
    errors: list[dict[str, object]] = []
    binary = records["gp_classification_binary"]
    binary_obj, binary_grad, _binary_iterations = expected_binary
    binary_values = np.array([real_token(binary[10]), real_token(binary[11])])
    binary_error = max(
        abs(real_token(binary[6]) - binary_obj),
        abs(real_token(binary[7]) - np.linalg.norm(binary_grad)),
        float(np.max(np.abs(binary_values - binary_grad))),
    )
    if binary_error > 3.0e-9 or np.any(binary_parameters < -5.0) or \
            np.any(binary_parameters > 5.0):
        raise RuntimeError(f"binary GP oracle mismatch: {binary_error:.3e}")
    if int(records["gp_classification_binary_invalid_bounds"][1]) != 1:
        raise RuntimeError("binary GP invalid-bound refusal was not reported")
    errors.append(row(
        details, workload="gp_classification_binary", phase="train", backend="fortml",
        status="pass", classes=2, repetitions=int(binary[3]),
        seconds_per_operation=float(binary[4]), metric="negative_mode_log_posterior",
        value=binary_obj, max_abs_error=binary_error,
        oracle="independent NumPy Laplace-mode and envelope-gradient recurrence",
        notes="log-variance/log-length bounds=[-5,5]; this is not full Laplace evidence",
    ))

    multi = records["gp_classification_multiclass"]
    objective = 0.0
    gradient = np.zeros(2)
    iterations = 0
    multi_parameters = np.array([real_token(multi[9]), real_token(multi[10])])
    for label in np.unique(MULTI):
        local = np.where(MULTI == label, 1, 0).astype(np.int64)
        local_objective, local_gradient, local_iterations = mode_objective(
            multi_parameters, local
        )
        objective += local_objective
        gradient += local_gradient
        iterations += local_iterations
    multi_error = max(
        abs(real_token(multi[7]) - objective),
        abs(real_token(multi[8]) - np.linalg.norm(gradient)),
        float(np.max(np.abs(np.array([real_token(multi[11]), real_token(multi[12])]) - gradient))),
    )
    if multi_error > 3.0e-8:
        raise RuntimeError(f"multiclass GP oracle mismatch: {multi_error:.3e}")
    errors.append(row(
        details, workload="gp_classification_multiclass", phase="train", backend="fortml",
        status="pass", classes=3, repetitions=int(multi[4]),
        seconds_per_operation=real_token(multi[5]), metric="negative_mode_log_posterior",
        value=objective, max_abs_error=multi_error,
        oracle="independent NumPy shared-kernel one-vs-rest mode recurrence",
        notes="shared log-kernel vector; summed envelope gradient; not full evidence",
    ))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/gp_classification_training.csv"))
    parser.add_argument("--target", default="fortml_bench_gp_classification_training")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    if args.skip_fortml:
        rows = [row(details, workload=name, phase="train", backend="fortml", status="skipped",
                    classes=classes, oracle="FortML release-app protocol", notes="--skip-fortml")
                for name, classes in (("gp_classification_binary", 2), ("gp_classification_multiclass", 3))]
    else:
        source = fortml / "app" / f"{args.target}.f90"
        if not source.is_file():
            rows = [row(details, workload="gp_classification", phase="train", backend="fortml",
                        status="unavailable", oracle="FortML release-app protocol",
                        notes=f"release target source is absent: {source.name}")]
        else:
            environment = os.environ.copy()
            environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
            subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                           check=True, capture_output=True, text=True)
            started = time.perf_counter()
            completed = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                                       env=environment, check=True, capture_output=True, text=True)
            if not completed.stdout:
                raise RuntimeError("GP release app emitted no records")
            rows = parse_app(completed.stdout, details)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
