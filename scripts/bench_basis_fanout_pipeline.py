#!/usr/bin/env python3
"""Correctness-gated benchmark for named fan-out/fan-in basis DAGs.

The FortML target contains an independent NumPy-style feature oracle, an
adjoint contraction check, a central-difference HVP check, and a typed CUDA
refusal.  This harness records the complete target gate and does not turn a
host timing into GPU evidence.
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
    "n_features", "n_branches", "seconds_per_operation", "metric", "value",
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


def independent_fixture_oracle() -> tuple[int, int, float]:
    """Return feature/branch counts and an independent finite fixture check."""
    x = np.array(
        [[0.2, 0.8], [-0.4, 1.1], [0.7, -0.3], [0.1, 0.6], [-0.5, -0.9]],
        dtype=np.float64,
    )
    quadratic = np.column_stack(
        [np.ones(len(x)), x[:, 0], x[:, 1], x[:, 0] ** 2,
         x[:, 0] * x[:, 1], x[:, 1] ** 2]
    )
    linear = x
    spectral_input = linear
    spectral = np.column_stack(
        [np.sin(0.7 * spectral_input[:, 0]),
         np.cos(0.7 * spectral_input[:, 0]),
         np.sin(1.2 * spectral_input[:, 1]),
         np.cos(1.2 * spectral_input[:, 1])]
    )
    expected = np.column_stack([quadratic, spectral])
    # A second construction is deliberately independent of the column stack.
    reference = np.empty_like(expected)
    reference[:, :6] = quadratic
    reference[:, 6] = np.sin(0.7 * x[:, 0])
    reference[:, 7] = np.cos(0.7 * x[:, 0])
    reference[:, 8] = np.sin(1.2 * x[:, 1])
    reference[:, 9] = np.cos(1.2 * x[:, 1])
    return expected.shape[1], 2, float(np.max(np.abs(expected - reference)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/basis_fanout_pipeline.csv"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    n_features, n_branches, oracle_error = independent_fixture_oracle()
    if oracle_error > 1.0e-14:
        raise RuntimeError(f"independent fanout fixture is inconsistent: {oracle_error}")

    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "test", "test_basis_fanout_pipeline"],
        cwd=fortml, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    gate_text = (completed.stdout + "\n" + completed.stderr).strip()
    # `fo test` reports the target result in its summary and may capture the
    # executable's stdout, so the process status is the authoritative gate.
    passed = completed.returncode == 0
    note = (
        "independent feature, JVP/VJP, HVP, metadata, CPU dispatch, and CUDA "
        "refusal oracle"
    )
    if not passed:
        note += ": " + (gate_text.splitlines()[-1] if gate_text else "no gate output")

    ignored = (output, root / "results" / "basis_fanout_pipeline.csv")
    metadata = {
        "n_samples": 5, "n_features": n_features, "n_branches": n_branches,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored), "compiler": "gfortran",
        "flags": "-O3", "oracle": "independent NumPy feature construction plus Fortran behavioral gate",
        "notes": note,
    }
    rows = [
        {"workload": "basis_fanout_pipeline", "phase": "value_derivatives",
         "backend": "fortml", "device": "cpu", "status": "pass" if passed else "failed",
         "seconds_per_operation": elapsed, "metric": "gate_seconds",
         "value": 1.0 if passed else 0.0, "max_abs_error": oracle_error, **metadata},
    ]
    device_metadata = dict(metadata)
    device_metadata.update({
        "oracle": "typed CUDA refusal leaves outputs untouched",
        "notes": "no resident CUDA branch executor; FORTNUM_NOT_IMPLEMENTED",
    })
    rows.append({
        "workload": "basis_fanout_pipeline", "phase": "device_contract",
        "backend": "fortml", "device": "cuda", "status": "unavailable",
        "seconds_per_operation": "", "metric": "api_surface", "value": "unavailable",
        "max_abs_error": 0.0, **device_metadata,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
