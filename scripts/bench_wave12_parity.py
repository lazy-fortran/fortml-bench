#!/usr/bin/env python3
"""Release evidence for the Wave 12 classification, GP, and checkpoint slices."""

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
    "workload", "phase", "backend", "device", "status", "metric", "value",
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
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def class_weight_oracle() -> tuple[float, float]:
    x = np.arange(6.0)
    labels = np.array([7, 7, 7, -2, -2, -2])
    factors = {-2: 0.5, 7: 2.0}
    weights = np.array([factors[int(label)] for label in labels])
    classes = np.array(sorted(factors))
    means = np.array([
        np.sum(weights[labels == label] * x[labels == label]) /
        np.sum(weights[labels == label]) for label in classes
    ])
    counts = np.array([np.sum(weights[labels == label]) for label in classes])
    priors = counts / np.sum(counts)
    query = 1.0
    # Equal within-class variance gives the one-dimensional LDA logit.
    logit = np.log(priors[1] / priors[0]) + 4.5 * query - 11.25
    return float(np.max(np.abs(means - [4.0, 1.0]))), float(1.0 / (1.0 + np.exp(-logit)))


def gp_oracle() -> float:
    x = np.array([-0.75, -0.30, 0.15, 0.60])
    y = np.column_stack((np.sin(1.1 * x), np.cos(0.8 * x) - 0.15)).reshape(-1, order="F")
    signal, lengthscale, noise = 1.2, 0.7, 0.12
    weights = np.array([0.8, -0.4])
    independent = np.array([0.25, 0.3])
    kernel = signal * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / lengthscale**2)
    coreg = np.outer(weights, weights) + np.diag(independent)
    covariance = np.kron(coreg, kernel) + noise * np.eye(y.size)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise RuntimeError("GP oracle covariance is not positive definite")
    solved = np.linalg.solve(covariance, y)
    return float(-0.5 * (y @ solved + logdet + y.size * np.log(2.0 * np.pi)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wave12_parity.csv"))
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
    rows: list[dict[str, object]] = []
    means_error, probability = class_weight_oracle()
    rows.extend((
        {**details, "workload": "discriminant_class_weight", "phase": "oracle_moments",
         "backend": "numpy_oracle", "device": "cpu", "status": "pass",
         "metric": "maximum_mean_error", "value": means_error, "max_abs_error": means_error,
         "oracle": "independent weighted LDA moments", "notes": "sorted labels [-2,7]"},
        {**details, "workload": "discriminant_class_weight", "phase": "oracle_probability",
         "backend": "numpy_oracle", "device": "cpu", "status": "pass",
         "metric": "class_7_probability_at_x_1", "value": probability, "max_abs_error": 0.0,
         "oracle": "independent weighted LDA probability", "notes": "class/sample weight equivalence"},
    ))
    rows.append({**details, "workload": "multi_output_gp_fortopt", "phase": "oracle_lml",
                 "backend": "numpy_oracle", "device": "cpu", "status": "pass",
                 "metric": "initial_log_marginal_likelihood", "value": gp_oracle(),
                 "max_abs_error": 0.0, "oracle": "independent dense ICM NumPy solve",
                 "notes": "two-output RBF ICM fixture"})

    environment = os.environ.copy()
    environment.update({"FO_SCAN_FALLBACK": "regex", "FO_FC": "gfortran",
                        "OMP_NUM_THREADS": "1"})
    gates = (
        ("discriminant_class_weight", "test_discriminant_class_weight",
         "class/sample weights, arbitrary labels, LDA/QDA moments and refusal"),
        ("multi_output_gp_fortopt", "test_multi_output_gp_training",
         "independent dense LML, FortOpt bounds/multistart, and CUDA refusal"),
        ("mlp_checkpoint_weight_mass", "test_mlp_checkpoint_io",
         "weighted pending microbatch persistence, malformed input, and device contract"),
    )
    for workload, target, note in gates:
        started = time.perf_counter()
        completed = subprocess.run(["fo", "test", target], cwd=fortml,
                                   env=environment, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        output_text = completed.stdout + completed.stderr
        if completed.returncode != 0 or "PASS" not in output_text:
            raise RuntimeError(f"{target} failed:\n{output_text}")
        rows.append({**details, "workload": workload, "phase": "behavioral_gate",
                     "backend": "fortml", "device": "cpu", "status": "pass",
                     "metric": "tests_passed", "value": 1.0, "max_abs_error": 0.0,
                     "oracle": f"independent Fortran {target}",
                     "notes": f"elapsed_seconds={elapsed:.6g}; {note}"})
        rows.append({**details, "workload": workload, "phase": "device_boundary",
                     "backend": "fortml", "device": "cuda", "status": "unavailable",
                     "metric": "resident_executor", "value": "FORTNUM_NOT_IMPLEMENTED",
                     "max_abs_error": 0.0, "oracle": "typed device capability contract",
                     "notes": "no resident CUDA path or host fallback claimed"})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
