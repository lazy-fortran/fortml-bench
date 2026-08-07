#!/usr/bin/env python3
"""Correctness gate for one-vs-rest variational GP multiclass prediction.

The independent NumPy fixture forms three latent posterior columns, applies a
Bernoulli probability link, normalises the positive columns to a simplex, and
checks the parameter JVP by finite differences.  FortML's release test covers
the corresponding inducing-point ELBO sum, packed gradient/JVP, sorted labels,
prediction JVP, malformed labels, and typed CUDA refusal.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_classes", "n_parameters", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[float, float, float, int]:
    """Return simplex error, probability-JVP FD error, sum, and dimension."""
    latent = np.array([
        [-1.2, -0.1, 0.8], [-0.7, 0.2, 0.4], [-0.2, 0.6, -0.1],
        [0.3, -0.4, 0.9], [0.8, 0.1, -0.6], [1.1, 0.5, -0.3],
    ], dtype=np.float64)
    direction = np.array([
        [0.02, -0.01, 0.03], [-0.01, 0.01, -0.02],
        [0.03, 0.02, 0.01], [0.01, -0.03, 0.02],
        [-0.02, 0.01, 0.01], [0.02, 0.02, -0.01],
    ])

    def probabilities(values: np.ndarray) -> np.ndarray:
        linked = 1.0 / (1.0 + np.exp(-values))
        return linked / np.sum(linked, axis=1, keepdims=True)

    linked = 1.0 / (1.0 + np.exp(-latent))
    normalized = probabilities(latent)
    simplex_error = float(np.max(np.abs(np.sum(normalized, axis=1) - 1.0)))
    linked_dot = linked * (1.0 - linked) * direction
    total = np.sum(linked, axis=1, keepdims=True)
    total_dot = np.sum(linked_dot, axis=1, keepdims=True)
    analytic = (linked_dot * total - linked * total_dot) / total**2
    step = 2.0e-6
    finite_difference = (probabilities(latent + step * direction) -
                         probabilities(latent - step * direction)) / (2.0 * step)
    jvp_error = float(np.max(np.abs(analytic - finite_difference)))
    if simplex_error > 2.0e-15 or jvp_error > 2.0e-10:
        raise RuntimeError(f"independent multiclass probability oracle failed: "
                           f"simplex={simplex_error:.3e}, jvp={jvp_error:.3e}")
    # Two inducing points use five packed variational coordinates per class
    # (two means, two diagonal logs, and one strict-lower entry).
    return simplex_error, jvp_error, float(np.sum(normalized)), 3 * 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_variational_multiclass_classification.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    simplex_error, jvp_error, probability_sum, n_parameters = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_gp_variational_multiclass_classification"],
                       cwd=fortml, check=True)
        status = "pass"
        notes = "FortML test covers OVR ELBO/gradient/JVP, sorted labels, prediction JVP"
    elapsed = time.perf_counter() - started
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
        row.update({"workload": "gp_variational_multiclass_classification",
                    "backend": "fortml", "device": "cpu", "n_samples": 6,
                    "n_classes": 3, "n_parameters": n_parameters})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass", metric="probability_simplex_sum",
        value=probability_sum, max_abs_error=max(simplex_error, jvp_error),
        oracle="independent NumPy OVR sigmoid normalization and JVP finite difference",
        notes=f"simplex_error={simplex_error:.3e}; jvp_error={jvp_error:.3e}")
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="probability_jvp_max_abs_error", value=jvp_error,
        max_abs_error=jvp_error,
        oracle="FortML test_gp_variational_multiclass_classification behavioral gate",
        notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_ovr_graph", value="nan", max_abs_error="nan",
        oracle="typed FortML CUDA capability refusal",
        notes="resident inducing solves and OVR reduction are not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
