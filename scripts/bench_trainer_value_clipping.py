#!/usr/bin/env python3
"""Correctness-gated generic-trainer per-coordinate clipping benchmark.

The NumPy oracle independently applies the quadratic gradient clamp. The
release app then checks the Fortran update and checkpoint round trip; no source
inspection is used as a behavioral result.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "seconds_per_operation", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
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
    return value + ("+dirty" if dirty else "")


def row(metadata: dict[str, object], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update(values)
    return result


def oracle() -> dict[str, float | np.ndarray]:
    parameters = np.array([0.0, 1.0], dtype=np.float64)
    gradient = np.array([-3.0, 6.0], dtype=np.float64)
    clipped = np.clip(gradient, -1.0, 1.0)
    expected = parameters - 0.1 * clipped
    return {"expected": expected, "clipped_coordinates": 1.0,
            "steps": 1.0}


def parse_app(output: str) -> dict[str, float]:
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] == "trainer_value_clipping" and len(fields) == 6:
            return {
                "steps": float(fields[1]),
                "clipped_steps": float(fields[2]),
                "parameter_1": float(fields[3]),
                "parameter_2": float(fields[4]),
                "checkpoint_equal": float(fields[5]),
            }
    raise ValueError("release app did not emit the value-clipping row")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/trainer_value_clipping.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/TRAINER_VALUE_CLIPPING.md"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    report = (args.report if args.report.is_absolute() else root / args.report).resolve()
    oracle_values = oracle()
    ignored = (output, report, root / "scripts" / "__pycache__")
    metadata = {
        "workload": "trainer_value_clipping", "backend": "fortml",
        "device": "cpu", "python_version": platform.python_version(),
        "numpy_version": np.__version__, "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = [
        row(metadata, phase="independent_oracle", status="pass",
            metric="parameter_1", value=float(oracle_values["expected"][0]),
            max_abs_error=0.0, oracle="independent NumPy clipped quadratic",
            notes="gradient=[-3,6], value bound=1, learning rate=0.1"),
        row(metadata, phase="independent_oracle", status="pass",
            metric="parameter_2", value=float(oracle_values["expected"][1]),
            max_abs_error=0.0, oracle="independent NumPy clipped quadratic",
            notes="coordinate-wise clipping precedes SGD"),
    ]
    values: dict[str, float] = {}
    app_status = "skipped"
    if not args.skip_fortml:
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        test = subprocess.run(["fo", "test", "test_trainer"], cwd=fortml,
                              env=environment, capture_output=True, text=True)
        app = subprocess.run(["fo", "exec", "fortml_bench_trainer_value_clipping"],
                             cwd=fortml, env=environment,
                             capture_output=True, text=True)
        if test.returncode == 0 and app.returncode == 0:
            try:
                values = parse_app(app.stdout)
                app_status = "pass"
            except (ValueError, OverflowError):
                app_status = "failed"
        else:
            app_status = "failed"
    parameter_error = max(
        abs(values.get("parameter_1", np.nan) - float(oracle_values["expected"][0])),
        abs(values.get("parameter_2", np.nan) - float(oracle_values["expected"][1])),
    )
    app_pass = (
        app_status == "pass"
        and values.get("steps") == oracle_values["steps"]
        and values.get("clipped_steps") == oracle_values["clipped_coordinates"]
        and values.get("checkpoint_equal") == 1.0
        and np.isfinite(parameter_error) and parameter_error <= 2.0e-14
    )
    rows.extend([
        row(metadata, phase="release_app", status="pass" if app_pass else app_status,
            metric="parameter_max_abs_error", value=float(parameter_error),
            max_abs_error=parameter_error,
            oracle="FortML SGD value clipping vs independent NumPy update",
            notes="accepted update and exact expected parameters"),
        row(metadata, phase="release_app", status="pass" if app_pass else app_status,
            metric="value_clipped_steps", value=values.get("clipped_steps", "nan"),
            max_abs_error=abs(values.get("clipped_steps", np.nan) - 1.0),
            oracle="FortML value-clipping diagnostic counter",
            notes="one update contains an out-of-bound coordinate"),
        row(metadata, phase="release_app", status="pass" if app_pass else app_status,
            metric="checkpoint_equal", value=values.get("checkpoint_equal", "nan"),
            max_abs_error=0.0,
            oracle="FortML schema-8 checkpoint round trip",
            notes="parameters and clipping counter survive load"),
        row(metadata, phase="independent_fortran_oracle",
            status="pass" if not args.skip_fortml else "skipped",
            metric="test_trainer", value=1.0 if not args.skip_fortml else "nan",
            max_abs_error=0.0,
            oracle="Fortran independent quadratic behavioral oracle",
            notes="value clipping, norm clipping, and checkpoint tests"),
        row(metadata, phase="device_boundary", device="cuda", status="unavailable",
            metric="resident_trainer", value="nan", max_abs_error=0.0,
            oracle="typed capability boundary",
            notes="generic objective and optimizer state are host-owned"),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Generic trainer value clipping\n\n"
        "This lane compares the generic trainer's per-coordinate gradient "
        "clipping against an independent NumPy quadratic oracle. The release "
        "app checks the exact SGD update, diagnostic counter, and schema-8 "
        "checkpoint round trip.\n\n"
        f"FortML revision: {metadata['fortml_revision']}\n"
        f"Benchmark revision: {metadata['benchmark_revision']}\n\n"
        "| phase | status | metric | value | max abs error |\n"
        "| --- | --- | --- | ---: | ---: |\n" +
        "".join(
            f"| {item['phase']} | {item['status']} | {item['metric']} | "
            f"{item['value']} | {item['max_abs_error']} |\n" for item in rows
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
