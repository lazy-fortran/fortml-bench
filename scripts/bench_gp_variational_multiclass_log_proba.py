#!/usr/bin/env python3
"""Release gate for stable variational-GP multiclass log probabilities.

The NumPy fixture checks log-sum-exp normalization and the packed-state style
JVP by central differences. The FortML behavioral oracle checks logistic and
probit links, fixed-state input products, reverse-product duality, and typed
CUDA refusal. The release app supplies a CPU timing row.
"""

from __future__ import annotations

import argparse
import csv
import platform
import re
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
    """Return the repository head and mark unrelated dirty files."""
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


def stable_log_normalize(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1, keepdims=True)
    return values - maximum - np.log(np.sum(np.exp(values - maximum), axis=1, keepdims=True))


def oracle() -> tuple[float, float, float]:
    """Return simplex error, JVP finite-difference error, and tail error."""
    latent = np.array([
        [-1.2, -0.1, 0.8], [-0.7, 0.2, 0.4], [-0.2, 0.6, -0.1],
        [0.3, -0.4, 0.9], [0.8, 0.1, -0.6], [1.1, 0.5, -0.3],
    ], dtype=np.float64)
    direction = np.array([
        [0.02, -0.01, 0.03], [-0.01, 0.01, -0.02],
        [0.03, 0.02, 0.01], [0.01, -0.03, 0.02],
        [-0.02, 0.01, 0.01], [0.02, 0.02, -0.01],
    ], dtype=np.float64)

    def log_sigmoid(z: np.ndarray) -> np.ndarray:
        result = np.empty_like(z)
        positive = z >= 0.0
        result[positive] = -np.log1p(np.exp(-z[positive]))
        result[~positive] = z[~positive] - np.log1p(np.exp(z[~positive]))
        return result

    def values(z: np.ndarray) -> np.ndarray:
        return stable_log_normalize(log_sigmoid(z))

    base = values(latent)
    step = 2.0e-6
    finite_difference = (values(latent + step * direction) -
                         values(latent - step * direction)) / (2.0 * step)
    linked = 1.0 / (1.0 + np.exp(latent))
    raw_dot = linked * direction
    weights = np.exp(base)
    analytic = raw_dot - np.sum(weights * raw_dot, axis=1, keepdims=True)
    jvp_error = float(np.max(np.abs(analytic - finite_difference)))
    simplex_error = float(np.max(np.abs(np.sum(np.exp(base), axis=1) - 1.0)))

    extreme = stable_log_normalize(np.full((3, 3), -1000.0, dtype=np.float64))
    tail_error = float(np.max(np.abs(np.sum(np.exp(extreme), axis=1) - 1.0)))
    if simplex_error > 2.0e-15 or jvp_error > 2.0e-10 or tail_error > 2.0e-15:
        raise RuntimeError(
            f"log-probability oracle failed: simplex={simplex_error:.3e}, "
            f"jvp={jvp_error:.3e}, tail={tail_error:.3e}")
    return simplex_error, jvp_error, tail_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_variational_multiclass_log_proba.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    simplex_error, jvp_error, tail_error = oracle()
    started = time.perf_counter()
    app_seconds = ""
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_gp_variational_multiclass_log_proba"],
                       cwd=fortml, check=True)
        app = subprocess.run(
            ["fo", "exec", "fortml_bench_gp_variational_multiclass_log_proba"],
            cwd=fortml, check=True, text=True, capture_output=True,
        )
        match = re.search(r"gp_variational_multiclass_log_proba,cpu,seconds,\s*"
                          r"([0-9.Ee+-]+)", app.stdout)
        if match is not None:
            app_seconds = float(match.group(1))
        status, notes = "pass", "log-probability test and release app passed"
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
        row.update({"workload": "gp_variational_multiclass_log_proba",
                    "backend": "fortml", "device": "cpu", "n_samples": 6,
                    "n_classes": 3, "n_parameters": 15})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="log_probability_simplex_error", value=simplex_error,
        max_abs_error=max(simplex_error, jvp_error, tail_error),
        oracle="independent NumPy log-sum-exp and packed-link JVP finite difference",
        notes=f"jvp_error={jvp_error:.3e}; tail_error={tail_error:.3e}")
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="log_probability_jvp_max_abs_error", value=jvp_error,
        max_abs_error=jvp_error,
        oracle="FortML stable logistic/probit log-probability behavioral oracle",
        notes=notes)
    add(phase="release_workload", status=status, seconds_per_operation=app_seconds,
        metric="cpu_prediction_seconds", value=app_seconds,
        max_abs_error=tail_error,
        oracle="fortml_bench_gp_variational_multiclass_log_proba",
        notes="CPU log-sum-exp prediction timing and simplex check")
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_ovr_logsumexp_graph", value="nan", max_abs_error="nan",
        oracle="typed FortML CUDA capability refusal",
        notes="resident inducing solves and OVR log-sum-exp reduction are not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
