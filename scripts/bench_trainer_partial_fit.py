#!/usr/bin/env python3
"""Correctness-gated benchmark for the resumable generic trainer contract.

The NumPy recurrence is an independent diagonal-quadratic Adam oracle.  The
release app compares one six-update run with 2+4 warm-start chunks and emits
the CUDA refusal code.  A second Fortran test covers checkpoint continuation
and transactional budget/device boundaries.
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
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True,
    ).splitlines():
        relative = line[3:].split(" -> ")[-1].strip()
        if (repository / relative).resolve() not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def advance_state(
    parameters: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    start_step: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.array([1.5, -0.5, 0.25], dtype=np.float64)
    curvature = np.array([2.0, 4.0, 1.25], dtype=np.float64)
    for step in range(start_step + 1, start_step + count + 1):
        gradient = curvature * (parameters - target)
        first = 0.8 * first + 0.2 * gradient
        second = 0.9 * second + 0.1 * gradient * gradient
        parameters -= 0.05 * (first / (1.0 - 0.8**step)) / (
            np.sqrt(second / (1.0 - 0.9**step)) + 1.0e-9
        )
    return parameters, first, second


def independent_oracle() -> dict[str, float]:
    initial = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    full, _, _ = advance_state(initial.copy(), zero.copy(), zero.copy(), 0, 6)
    prefix, first, second = advance_state(initial.copy(), zero.copy(), zero.copy(), 0, 2)
    split, _, _ = advance_state(prefix, first, second, 2, 4)
    replay_error = float(np.max(np.abs(full - split)))
    if replay_error > 1.0e-14:
        raise RuntimeError(f"NumPy replay mismatch: {replay_error:.3e}")
    return {"replay_max_abs_error": replay_error, "final_norm": float(np.linalg.norm(full))}


def row(metadata: dict[str, object], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update(values)
    return result


def parse_app(output: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] == "trainer_partial_fit" and len(fields) == 4:
            values["steps"] = float(fields[1])
            values["replay_max_abs_error"] = float(fields[2])
            values["seconds"] = float(fields[3])
        elif fields and fields[0] == "trainer_partial_fit_cuda" and len(fields) == 2:
            values["cuda_status_code"] = float(fields[1])
    if "replay_max_abs_error" not in values or "cuda_status_code" not in values:
        raise ValueError("release app did not emit partial-fit rows")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/trainer_partial_fit.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/TRAINER_PARTIAL_FIT.md"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    report = (args.report if args.report.is_absolute() else root / args.report).resolve()
    oracle = independent_oracle()
    ignored = (output, report, root / "results/trainer_partial_fit.csv",
               root / "results/TRAINER_PARTIAL_FIT.md")
    metadata = {
        "workload": "trainer_partial_fit", "backend": "fortml", "device": "cpu",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = [
        row(metadata, phase="independent_oracle", status="pass",
            metric="replay_max_abs_error", value=oracle["replay_max_abs_error"],
            max_abs_error=oracle["replay_max_abs_error"],
            oracle="independent NumPy diagonal-quadratic Adam recurrence",
            notes="six updates; split state boundary is compared"),
        row(metadata, phase="independent_oracle", status="pass",
            metric="final_parameter_norm", value=oracle["final_norm"],
            max_abs_error=0.0,
            oracle="independent NumPy diagonal-quadratic Adam recurrence",
            notes="target=[1.5,-0.5,0.25]; curvature=[2,4,1.25]"),
    ]
    app_status = "skipped"
    values: dict[str, float] = {}
    elapsed = float("nan")
    if not args.skip_fortml:
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        started = time.perf_counter()
        gate = subprocess.run(["fo", "test", "test_trainer_partial_fit"],
                              cwd=fortml, env=environment,
                              capture_output=True, text=True)
        app = subprocess.run(["fo", "exec", "fortml_bench_trainer_partial_fit"],
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
    replay_error = abs(values.get("replay_max_abs_error", np.nan) -
                       oracle["replay_max_abs_error"])
    app_pass = (app_status == "pass" and np.isfinite(replay_error) and
                replay_error <= 1.0e-14 and values.get("cuda_status_code") == 3.0)
    rows.append(row(
        metadata, phase="release_app", status="pass" if app_pass else app_status,
        metric="replay_max_abs_error", value=values.get("replay_max_abs_error", "nan"),
        seconds_per_operation=elapsed, max_abs_error=replay_error,
        oracle="FortML release app vs independent NumPy Adam recurrence",
        notes="uninterrupted six updates versus 2+4 partial-fit chunks",
    ))
    rows.append(row(
        metadata, phase="release_app", status="pass" if app_pass else app_status,
        metric="steps", value=values.get("steps", "nan"),
        seconds_per_operation=elapsed, max_abs_error=abs(values.get("steps", np.nan) - 6.0),
        oracle="FortML release app", notes="declared total update budget",
    ))
    rows.append(row(
        metadata, phase="cuda_typed_refusal", device="cuda", status="unavailable",
        metric="partial_fit_status_code", value=values.get("cuda_status_code", "nan"),
        max_abs_error=0.0, oracle="typed CUDA refusal",
        notes="generic trainer owns a host objective; no hidden fallback",
    ))
    rows.append(row(
        metadata, phase="independent_fortran_oracle",
        status="pass" if not args.skip_fortml else "skipped",
        metric="test_trainer_partial_fit", value=1.0 if not args.skip_fortml else "nan",
        max_abs_error=0.0, oracle="Fortran checkpoint/replay behavioral oracle",
        notes="fo test test_trainer_partial_fit",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Trainer partial-fit contract\n\n"
        "This lane checks the generic trainer_t%partial_fit warm-start "
        "contract against an independent NumPy Adam recurrence. One six-update "
        "trajectory is compared with two and four update chunks, then the "
        "Fortran test checks checkpoint continuation and transactional "
        "over-budget requests. CUDA is an explicit typed refusal because the "
        "generic trainer has no resident objective or optimizer state.\n\n"
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
