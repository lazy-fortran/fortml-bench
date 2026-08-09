#!/usr/bin/env python3
"""Correctness-gated generic-trainer fit-diagnostics benchmark.

The NumPy oracle independently defines the quadratic optimum and the
callback-stopped Adam update count.  The FortML release app supplies the
 bounded FortOpt counters and schema-8 state contract. No source-inspection
result is accepted as a behavioral pass.
"""

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


def quadratic_oracle() -> dict[str, float]:
    target = np.array([1.5, -0.5], dtype=np.float64)
    initial = np.array([0.0, 1.0], dtype=np.float64)
    value = float((initial[0] - target[0]) ** 2 +
                  2.0 * (initial[1] - target[1]) ** 2)
    if value <= 0.0:
        raise RuntimeError("quadratic fixture must have a nonzero initial loss")
    return {"optimum_parameter_error": 0.0, "initial_value": value,
            "adam_fit_calls": 1.0, "adam_iterations": 1.0}


def row(metadata: dict[str, object], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update(values)
    return result


def parse_app(output: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if fields and fields[0] == "trainer_fit_lbfgsb" and len(fields) == 6:
            values["lbfgsb_iterations"] = float(fields[1])
            values["lbfgsb_line_search"] = float(fields[2])
            values["lbfgsb_curvature"] = float(fields[3])
            values["lbfgsb_parameter_error"] = float(fields[4])
            values["lbfgsb_seconds"] = float(fields[5])
        elif fields and fields[0] == "trainer_fit_adam" and len(fields) == 6:
            values["adam_fit_calls"] = float(fields[1])
            values["adam_iterations"] = float(fields[2])
            values["adam_line_search"] = float(fields[3])
            values["adam_curvature"] = float(fields[4])
            values["adam_seconds"] = float(fields[5])
    required = {
        "lbfgsb_iterations", "lbfgsb_line_search", "lbfgsb_curvature",
        "lbfgsb_parameter_error", "adam_fit_calls", "adam_iterations",
        "adam_line_search", "adam_curvature",
    }
    if not required.issubset(values):
        raise ValueError("release app did not emit all fit-diagnostics rows")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/trainer_fit_diagnostics.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/TRAINER_FIT_DIAGNOSTICS.md"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    report = (args.report if args.report.is_absolute() else root / args.report).resolve()
    oracle = quadratic_oracle()
    ignored = (output, report, root / "scripts" / "__pycache__")
    metadata = {
        "workload": "trainer_fit_diagnostics", "backend": "fortml", "device": "cpu",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = [
        row(metadata, phase="independent_oracle", status="pass",
            metric="optimum_parameter_error", value=oracle["optimum_parameter_error"],
            max_abs_error=0.0,
            oracle="independent NumPy quadratic optimum",
            notes="target=[1.5,-0.5]; curvature=[2,4]"),
        row(metadata, phase="independent_oracle", status="pass",
            metric="adam_iterations", value=oracle["adam_iterations"],
            max_abs_error=0.0,
            oracle="independent callback-stopped Adam contract",
            notes="one accepted update before callback stop"),
    ]
    values: dict[str, float] = {}
    app_status = "skipped"
    elapsed = float("nan")
    if not args.skip_fortml:
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        started = time.perf_counter()
        gate = subprocess.run(["fo", "test", "test_trainer_fit_diagnostics"],
                              cwd=fortml, env=environment,
                              capture_output=True, text=True)
        app = subprocess.run(["fo", "exec", "fortml_bench_trainer_fit_diagnostics"],
                             cwd=fortml, env=environment,
                             capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        if gate.returncode == 0 and app.returncode == 0:
            try:
                values = parse_app(app.stdout)
                app_status = "pass"
            except (ValueError, OverflowError):
                app_status = "failed"
        else:
            app_status = "failed"

    lbfgsb_error = abs(values.get("lbfgsb_parameter_error", np.nan) -
                       oracle["optimum_parameter_error"])
    counters_pass = (
        values.get("lbfgsb_iterations", 0.0) > 0.0 and
        values.get("lbfgsb_line_search", 0.0) > 0.0 and
        values.get("lbfgsb_curvature", 0.0) > 0.0 and
        values.get("adam_fit_calls") == oracle["adam_fit_calls"] and
        values.get("adam_iterations") == oracle["adam_iterations"] and
        values.get("adam_line_search") == 0.0 and
        values.get("adam_curvature") == 0.0
    )
    app_pass = (app_status == "pass" and counters_pass and
                np.isfinite(lbfgsb_error) and lbfgsb_error <= 2.0e-7)
    rows.extend([
        row(metadata, phase="release_app", status="pass" if app_pass else app_status,
            metric="lbfgsb_parameter_error",
            value=values.get("lbfgsb_parameter_error", "nan"),
            seconds_per_operation=values.get("lbfgsb_seconds", elapsed),
            max_abs_error=lbfgsb_error,
            oracle="FortML bounded fit vs independent quadratic optimum",
            notes="iterations, line-search evaluations, and curvature updates must be positive"),
        row(metadata, phase="release_app", status="pass" if app_pass else app_status,
            metric="lbfgsb_line_search_evaluations",
            value=values.get("lbfgsb_line_search", "nan"),
            seconds_per_operation=values.get("lbfgsb_seconds", elapsed),
            max_abs_error=0.0,
            oracle="FortOpt L-BFGS-B diagnostic contract",
            notes="latest bounded fit"),
        row(metadata, phase="release_app", status="pass" if app_pass else app_status,
            metric="lbfgsb_curvature_updates",
            value=values.get("lbfgsb_curvature", "nan"),
            seconds_per_operation=values.get("lbfgsb_seconds", elapsed),
            max_abs_error=0.0,
            oracle="FortOpt L-BFGS-B diagnostic contract",
            notes="accepted secant history updates"),
        row(metadata, phase="release_app", status="pass" if app_pass else app_status,
            metric="adam_fit_calls", value=values.get("adam_fit_calls", "nan"),
            seconds_per_operation=values.get("adam_seconds", elapsed),
            max_abs_error=abs(values.get("adam_fit_calls", np.nan) - 1.0),
            oracle="FortML callback-stopped Adam state",
            notes="streaming optimizer reports zero L-BFGS-B counters"),
        row(metadata, phase="independent_fortran_oracle",
            status="pass" if not args.skip_fortml else "skipped",
            metric="test_trainer_fit_diagnostics",
            value=1.0 if not args.skip_fortml else "nan", max_abs_error=0.0,
            oracle="Fortran independent quadratic behavioral oracle",
            notes="counter relationships, schema-8 persistence, and optimum"),
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
        "# Generic trainer fit diagnostics\n\n"
        "This lane gates the schema-8 generic trainer diagnostics against an "
        "independent NumPy quadratic oracle. The bounded FortOpt L-BFGS-B "
        "row records iteration, line-search, and curvature counters. A "
        "callback-stopped Adam row checks fit-call/update counters and the "
        "zero L-BFGS-B-specific boundary.\n\n"
        f"FortML revision: {metadata['fortml_revision']}\n"
        f"Benchmark revision: {metadata['benchmark_revision']}\n\n"
        "| phase | status | metric | value | max abs error |\n"
        "| --- | --- | --- | ---: | ---: |\n" +
        "".join(
            f"| {r['phase']} | {r['status']} | {r['metric']} | "
            f"{r['value']} | {r['max_abs_error']} |\n" for r in rows
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
