#!/usr/bin/env python3
"""Correctness-gated dense k-means benchmark.

The NumPy implementation is an independent oracle for the exact deterministic
fixture and seeded cyclic initialization used by the FortML release app.  The
FortML timing is retained only after its reported inertia agrees with that
oracle.  CUDA is recorded as an explicit unavailable row because the current
estimator contract refuses device-resident execution rather than falling back
to the host.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 240
N_FEATURES = 2
N_CLUSTERS = 3
SEED = 7
MAX_ITER = 100
TOLERANCE = 1.0e-8
REPETITIONS = 8
INERTIA_TOLERANCE = 1.0e-10

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_clusters", "seed", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
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


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, root / "results" / "kmeans.csv")),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def fixture() -> np.ndarray:
    values = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    for index in range(80):
        offset = index / 80.0
        values[index] = (-0.2 + 0.4 * offset, 0.1 * np.sin(offset))
        values[80 + index] = (5.0 + 0.4 * offset, 5.1 * np.cos(offset))
        values[160 + index] = (10.0 - 0.4 * offset, 0.2 + 0.1 * np.sin(offset))
    return values


def numpy_kmeans(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centers = x[(SEED - 1 + np.arange(N_CLUSTERS)) % len(x)].copy()
    for _ in range(MAX_ITER):
        squared = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(squared, axis=1)
        counts = np.bincount(labels, minlength=N_CLUSTERS)
        if np.any(counts == 0):
            raise RuntimeError("NumPy oracle encountered an empty cluster")
        next_centers = np.vstack([
            x[labels == cluster].mean(axis=0) for cluster in range(N_CLUSTERS)
        ])
        shift = float(np.max(np.sqrt(((next_centers - centers) ** 2).sum(axis=1))))
        centers = next_centers
        if shift <= TOLERANCE:
            break
    squared = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(squared, axis=1)
    counts = np.bincount(labels, minlength=N_CLUSTERS)
    if np.any(counts == 0):
        raise RuntimeError("NumPy final assignment has an empty cluster")
    return centers, labels, float(squared[np.arange(len(x)), labels].sum())


def base_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict(details)
    row.update({
        "workload": "kmeans", "phase": "", "backend": "", "device": "cpu",
        "status": "", "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_clusters": N_CLUSTERS, "seed": SEED, "repetitions": "",
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               expected_inertia: float) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return [base_row(details, phase="fit", backend="fortml", status="unavailable",
                         oracle="FortML release-app protocol",
                         notes=f"release target source is absent: {source.name}")]
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
        return [base_row(details, phase="fit", backend="fortml", status="unavailable",
                         oracle="FortML release-app protocol", notes=note)]
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                         env=environment, capture_output=True, text=True)
    if run.returncode != 0:
        note = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "release app failed"
        return [base_row(details, phase="fit", backend="fortml", status="unavailable",
                         oracle="FortML release-app protocol", notes=note)]
    patterns = {
        "fit": r"^kmeans_fit_seconds_per_operation,\s*([0-9Ee+.-]+)$",
        "transform": r"^kmeans_transform_seconds_per_operation,\s*([0-9Ee+.-]+)$",
        "inertia": r"^kmeans_inertia,\s*([0-9Ee+.-]+)$",
    }
    parsed: dict[str, float] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, run.stdout, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"FortML k-means {name} record not found: {run.stdout!r}")
        parsed[name] = float(match.group(1))
    inertia_error = abs(parsed["inertia"] - expected_inertia)
    if inertia_error > INERTIA_TOLERANCE:
        raise RuntimeError(f"FortML k-means inertia mismatch: {inertia_error:.3e}")
    oracle = "independent NumPy seeded cyclic Lloyd oracle"
    return [
        base_row(details, phase="fit", backend="fortml", status="pass",
                 repetitions=REPETITIONS, seconds_per_operation=parsed["fit"],
                 metric="inertia", value=parsed["inertia"], max_abs_error=inertia_error,
                 oracle=oracle, notes=target),
        base_row(details, phase="transform", backend="fortml", status="pass",
                 repetitions=REPETITIONS, seconds_per_operation=parsed["transform"],
                 metric="inertia", value=parsed["inertia"], max_abs_error=inertia_error,
                 oracle=oracle, notes="fixed-center Euclidean distances"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/kmeans.csv"))
    parser.add_argument("--target", default="fortml_bench_kmeans")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, output)
    x = fixture()
    started = time.perf_counter()
    expected_centers, expected_labels, expected_inertia = numpy_kmeans(x)
    oracle_seconds = time.perf_counter() - started
    if expected_centers.shape != (N_CLUSTERS, N_FEATURES) or expected_labels.shape != (N_SAMPLES,):
        raise RuntimeError("NumPy k-means oracle shape check failed")
    rows = [base_row(details, phase="fit", backend="numpy_oracle", status="pass",
                     repetitions=1, seconds_per_operation=oracle_seconds,
                     metric="inertia", value=expected_inertia, max_abs_error=0.0,
                     oracle="independent seeded cyclic Lloyd implementation",
                     notes="center and label arrays checked internally")]
    if args.skip_fortml:
        rows.extend([base_row(details, phase="fit", backend="fortml", status="skipped",
                              oracle="FortML release-app protocol", notes="--skip-fortml")])
    else:
        rows.extend(run_fortml(fortml, args.target, details, expected_inertia))
    rows.append(base_row(details, phase="fit", backend="fortml", device="cuda",
                         status="unavailable", metric="inertia", oracle="typed device contract",
                         notes="CUDA fit returns FORTNUM_NOT_IMPLEMENTED; no host fallback"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

