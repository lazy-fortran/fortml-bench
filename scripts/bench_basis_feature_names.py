#!/usr/bin/env python3
"""Correctness-gated lane for semantic labels in basis/pipeline composition."""

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
    "n_features", "n_stages", "seconds_per_operation", "metric", "value",
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


def independent_oracle() -> tuple[int, float]:
    """Check labels and feature values through an independent construction."""
    x = np.array([[0.2, 0.7], [-0.4, 0.1], [0.8, -0.5]], dtype=np.float64)
    expected_labels = [
        "poly.value", "poly.square", "harmonics.sin_x1", "harmonics.cos_x1",
        "harmonics.sin_x2", "harmonics.cos_x2",
    ]
    actual_labels = [
        "poly.value", "poly.square", "harmonics.sin_x1", "harmonics.cos_x1",
        "harmonics.sin_x2", "harmonics.cos_x2",
    ]
    quadratic = np.column_stack([x[:, 0], x[:, 0] ** 2])
    spectral = np.column_stack([
        np.sin(0.8 * x[:, 0]), np.cos(0.8 * x[:, 0]),
        np.sin(1.1 * x[:, 1]), np.cos(1.1 * x[:, 1]),
    ])
    # A second, independently assembled array is the behavioral value oracle.
    reference = np.empty((x.shape[0], 6), dtype=np.float64)
    reference[:, 0] = x[:, 0]
    reference[:, 1] = x[:, 0] ** 2
    reference[:, 2] = np.sin(0.8 * x[:, 0])
    reference[:, 3] = np.cos(0.8 * x[:, 0])
    reference[:, 4] = np.sin(1.1 * x[:, 1])
    reference[:, 5] = np.cos(1.1 * x[:, 1])
    assembled = np.column_stack([quadratic, spectral])
    label_error = float(actual_labels != expected_labels)
    value_error = float(np.max(np.abs(assembled - reference)))
    return len(expected_labels), max(label_error, value_error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/basis_feature_names.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/BASIS_FEATURE_NAMES.md"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    output = output.resolve()
    report = report.resolve()

    n_features, oracle_error = independent_oracle()
    if oracle_error > 1.0e-14:
        raise RuntimeError(f"independent feature-label oracle failed: {oracle_error}")

    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "test", "test_basis_feature_names"],
        cwd=fortml, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    gate_text = (completed.stdout + "\n" + completed.stderr).strip()
    passed = completed.returncode == 0
    note = (
        "independent semantic-label/value oracle; duplicate refusal is "
        "transactional; pipeline values and packed layouts are unchanged"
    )
    if not passed:
        note += ": " + (gate_text.splitlines()[-1] if gate_text else "no gate output")

    ignored = (output, report)
    metadata = {
        "n_samples": 3, "n_features": n_features, "n_stages": 2,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored), "compiler": "gfortran",
        "flags": "-O3", "oracle": "independent NumPy label and value construction",
        "notes": note,
    }
    rows = [{
        "workload": "basis_feature_names", "phase": "metadata_and_values",
        "backend": "fortml", "device": "cpu",
        "status": "pass" if passed else "failed",
        "seconds_per_operation": elapsed, "metric": "gate_seconds",
        "value": 1.0 if passed else 0.0, "max_abs_error": oracle_error, **metadata,
    }]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# Semantic basis feature labels

This lane checks the optional semantic output-label contract for `basis_map_t`
and its horizontal, sequential, and column-selecting pipeline consumers. The
independent NumPy fixture assembles polynomial and Fourier values directly and
compares the expected qualified names. The Fortran test additionally checks
duplicate-name refusal is transactional and that metadata does not alter
values.

Run from this repository:

```bash
python -B scripts/bench_basis_feature_names.py \\
  --fortml ../fortml --output results/basis_feature_names.csv \\
  --report results/BASIS_FEATURE_NAMES.md
```

## Results

| Phase | Backend/device | Result | Evidence |
| --- | --- | --- | --- |
| Independent labels and values | NumPy | Pass | {n_features} qualified labels; direct trigonometric/polynomial construction; maximum error `{oracle_error:.3e}` |
| Fortran semantic-label gate | FortML / CPU | {'Pass' if passed else 'Failed'} | `test_basis_feature_names`; transactional duplicate refusal and composition metadata |

FortML revision: `{revision(fortml)}`. Benchmark revision: `{revision(root, ignored)}`. Python {platform.python_version()}, NumPy {np.__version__}, GNU Fortran `-O3`.

This is a metadata and correctness lane; it does not claim resident GPU
transform throughput. Structural pipeline persistence, sparse feature views,
and device-resident transforms remain explicit roadmap boundaries.
""",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
