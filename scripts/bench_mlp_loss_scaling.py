#!/usr/bin/env python3
"""Correctness-gated MLP automatic loss-scaling benchmark.

The NumPy recurrence is independent of FortML.  The release app reports the
same growth and overflow transitions, the FP64 trainer state, and the typed
FP32 boundary.  CUDA is recorded as unavailable because resident mixed-
precision kernels are not part of the current contract.
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
            text=True).splitlines():
        relative = line[3:].split(" -> ")[-1].strip()
        path = (repository / relative).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def independent_recurrence() -> dict[str, float]:
    scale = 8.0
    good_steps = 0
    overflow_count = 0
    skipped_updates = 0
    for finite, applied in ((True, True), (True, True), (False, False)):
        if not finite:
            scale = max(1.0, scale * 0.5)
            good_steps = 0
            overflow_count += 1
            skipped_updates += 1
        elif applied:
            good_steps += 1
            if good_steps >= 2:
                scale = min(32.0, scale * 2.0)
                good_steps = 0
    return {
        "final_scale": scale,
        "good_steps": float(good_steps),
        "overflow_count": float(overflow_count),
        "skipped_updates": float(skipped_updates),
    }


def app_values(output: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if fields and fields[0] in {
                "recurrence_overflow", "fp64_training", "fp32_typed_refusal"}:
            values[fields[0]] = [float(item) for item in fields[1:]]
    return values


def row(metadata: dict[str, object], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_loss_scaling.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/MLP_LOSS_SCALING.md"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    report = args.report if args.report.is_absolute() else root / args.report
    report = report.resolve()
    expected = independent_recurrence()
    ignored = (output, report, root / "results/mlp_loss_scaling.csv")
    metadata = {
        "workload": "mlp_loss_scaling",
        "backend": "fortml",
        "device": "cpu",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(
            fortml, (fortml / "test_mlp_loss_scaling_checkpoint.txt",)),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = []
    rows.extend(
        row(metadata, phase="independent_recurrence", status="pass",
            metric=key, value=value, max_abs_error=0.0,
            oracle="independent NumPy growth/backoff recurrence",
            notes="initial=8, growth=2, backoff=0.5, interval=2")
        for key, value in expected.items()
    )
    app_result = None
    elapsed = float("nan")
    if not args.skip_fortml:
        started = time.perf_counter()
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        app_result = subprocess.run(
            ["fo", "exec", "fortml_bench_mlp_loss_scaling"], cwd=fortml,
            env=environment, capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - started
    if app_result is None:
        parsed: dict[str, list[float]] = {}
        app_status = "skipped"
        note = "--skip-fortml"
    else:
        parsed = app_values(app_result.stdout)
        app_status = "pass" if app_result.returncode == 0 else "failed"
        note = "release app recurrence and typed refusal rows"
        if app_result.returncode != 0:
            note = (app_result.stderr[-240:] or app_result.stdout[-240:]).replace("\n", " ")
    observed_overflow = parsed.get("recurrence_overflow", [float("nan")]*3)
    observed_training = parsed.get("fp64_training", [float("nan")]*3)
    observed_refusal = parsed.get("fp32_typed_refusal", [float("nan")])
    errors = np.array([
        abs(observed_overflow[0] - expected["final_scale"]),
        abs(observed_overflow[1] - expected["overflow_count"]),
        abs(observed_overflow[2] - expected["skipped_updates"]),
        abs(observed_training[1] - 16.0),
        abs(observed_refusal[0] - 3.0),
    ])
    app_passed = app_status == "pass" and np.all(np.isfinite(errors)) and np.max(errors) == 0.0
    rows.append(row(metadata, phase="release_app_recurrence", status="pass" if app_passed else app_status,
                    metric="final_scale", value=observed_overflow[0],
                    seconds_per_operation=elapsed, max_abs_error=float(np.max(errors)),
                    oracle="FortML app vs independent NumPy recurrence", notes=note))
    rows.append(row(metadata, phase="fp64_training_state", status=app_status,
                    metric="loss_scale", value=observed_training[1],
                    max_abs_error=abs(observed_training[1] - 16.0),
                    oracle="FP64 trainer captures dynamic scale and counters",
                    notes=f"updates={observed_training[0] if observed_training else 'nan'}"))
    rows.append(row(metadata, phase="fp32_typed_refusal", status=app_status,
                    metric="status_code", value=observed_refusal[0],
                    max_abs_error=abs(observed_refusal[0] - 3.0),
                    oracle="FORTNUM_NOT_IMPLEMENTED before model mutation",
                    notes="master weights and lower-precision resident kernels remain open"))
    rows.append(row(metadata, phase="cuda_typed_refusal", device="cuda",
                    status="unavailable", metric="resident_loss_scaling", value="nan",
                    max_abs_error=0.0, oracle="typed CUDA boundary",
                    notes="no hidden host fallback"))
    if not args.skip_fortml:
        test = subprocess.run(
            ["fo", "test", "test_mlp_loss_scaling"], cwd=fortml,
            env=environment, capture_output=True, text=True,
        )
        test_status = "pass" if test.returncode == 0 else "failed"
        rows.append(row(metadata, phase="independent_fortran_oracle", status=test_status,
                        metric="test_mlp_loss_scaling", value=1.0 if test.returncode == 0 else 0.0,
                        max_abs_error=0.0,
                        oracle="Fortran recurrence/checkpoint/refusal oracle",
                        notes="fo test test_mlp_loss_scaling"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# MLP loss scaling",
        "",
        "The lane compares the release app with an independent NumPy recurrence.",
        "The policy starts at 8, grows by 2 after two finite updates, and backs",
        "off by 0.5 after an overflow. The FP64 trainer row checks persisted",
        "dynamic state. FP32 and CUDA rows record typed capability boundaries.",
        "",
        f"FortML revision: `{metadata['fortml_revision']}`  ",
        f"Benchmark revision: `{metadata['benchmark_revision']}`",
        "",
        "| phase | status | metric | value | max abs error |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for item in rows:
        report_lines.append(
            f"| {item['phase']} | {item['status']} | {item['metric']} | "
            f"{item['value']} | {item['max_abs_error']} |"
        )
    report.write_text("\n".join(report_lines) + "\n")
    print(f"wrote {len(rows)} rows to {output}; report={report}")
    if not args.skip_fortml and not app_passed:
        print((app_result.stdout if app_result else "") + (app_result.stderr if app_result else ""))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
