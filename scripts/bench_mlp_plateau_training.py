#!/usr/bin/env python3
"""Correctness-gated metric-aware plateau trainer benchmark.

The schedule recurrence is evaluated independently in NumPy. The Fortran
release app supplies the CPU trainer result and wall time; CUDA is deliberately
recorded as a typed unavailable capability until metric reduction and optimizer
state can remain resident.
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
    "workload", "phase", "backend", "device", "status", "n_parameters",
    "epochs", "updates", "seconds_per_operation", "metric", "value",
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
        path = (repository / line[3:].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def expected_rate() -> float:
    base = 1.0e-12
    factor = 0.4
    patience = 2
    min_delta = 0.02
    best = 0.5 * np.mean((2.0 * np.array([-1.0, 0.0, 1.0])) ** 2)
    bad = 0
    reductions = 0
    for _ in range(4):
        metric = best
        improved = metric < best - min_delta
        if improved:
            best = metric
            bad = 0
        elif bad >= patience - 1:
            bad = 0
            reductions += 1
        else:
            bad += 1
    return base * factor**reductions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_plateau_training.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_plateau_training")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    expected = expected_rate()
    started = time.perf_counter()
    run = subprocess.run(
        ["fo", "exec", args.target], cwd=fortml, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    output_text = (run.stdout + "\n" + run.stderr).strip()
    final_rate = None
    updates = None
    for line in output_text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4 and fields[0] == "mlp_plateau_training":
            if fields[2] == "final_learning_rate":
                final_rate = float(fields[3])
            elif fields[2] == "updates":
                updates = int(fields[3])
    passed = (
        run.returncode == 0 and final_rate is not None and updates == 4 and
        abs(final_rate - expected) <= 1.0e-24
    )
    ignored = (output, root / "results" / "mlp_plateau_training.csv")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored), "compiler": "gfortran",
        "flags": "-O3",
    }
    rows = [{
        "workload": "mlp_plateau_training", "phase": "trainer",
        "backend": "fortml", "device": "cpu", "status": "pass" if passed else "failed",
        "n_parameters": 2, "epochs": 4, "updates": updates or "",
        "seconds_per_operation": elapsed / 4.0,
        "metric": "final_learning_rate", "value": final_rate or "",
        "max_abs_error": abs((final_rate or 0.0) - expected) if final_rate is not None else "",
        "oracle": "independent NumPy plateau state recurrence", **details,
        "notes": "patience=2; min_delta=0.02; factor=0.4; CPU trainer",
    }, {
        "workload": "mlp_plateau_training", "phase": "device_boundary",
        "backend": "fortml", "device": "cuda", "status": "unavailable",
        "n_parameters": 2, "epochs": 4, "updates": "",
        "seconds_per_operation": "", "metric": "typed_refusal", "value": "",
        "max_abs_error": "", "oracle": "FortML device capability contract", **details,
        "notes": "metric reduction and optimizer state are not resident; no host fallback",
    }]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    if not passed:
        print(output_text)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
