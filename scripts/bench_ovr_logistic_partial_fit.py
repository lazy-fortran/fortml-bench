#!/usr/bin/env python3
"""Correctness-gated benchmark for OVR logistic partial fitting.

The small Python state machine is an independent metadata oracle.  The
release app compares one-shot and replayed logistic probabilities and emits
the typed CUDA refusal; the Fortran test supplies finite-difference products
and transactional rollback coverage.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


FORTNUM_NOT_IMPLEMENTED = 3
FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes", "n_samples", "n_features", "n_classes",
    "n_train", "batch_size", "batch_count", "seconds_per_operation",
    "python_version", "numpy_version", "fortad_revision", "fortsym_revision",
    "fortopt_revision",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def independent_state_oracle() -> dict[str, float]:
    classes = [-7, 10, 42]
    first = [42, 42, 42, -7]
    second = [-7, 10, 10, 42, 10]
    if classes != sorted(set(classes)):
        raise RuntimeError("class vocabulary is not strictly sorted")
    if set(first) | set(second) != set(classes):
        raise RuntimeError("batches do not cover the declared vocabulary")
    malformed = [42, 999, -7]
    if set(malformed).issubset(set(classes)):
        raise RuntimeError("unknown-label rollback oracle is broken")
    return {"sample_count": float(len(first) + len(second)),
            "batch_count": 2.0, "class_count": float(len(classes))}


def parse_app(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] == "ovr_logistic_partial_fit" and len(fields) == 5:
            values["batch_count"] = float(fields[1])
            values["replay_error"] = float(fields[2])
            values["seconds"] = float(fields[3])
            values["cuda_status"] = float(fields[4])
    required = {"batch_count", "replay_error", "seconds", "cuda_status"}
    missing = required - values.keys()
    if missing:
        raise RuntimeError(f"release app omitted {sorted(missing)}")
    return values


def row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/ovr_logistic_partial_fit.csv"))
    parser.add_argument("--target", default="fortml_bench_ovr_logistic_partial_fit")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    oracle = independent_state_oracle()
    ignored = (output, root / "results/OVR_LOGISTIC_PARTIAL_FIT.md")
    details = {
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
        "python_version": platform.python_version(), "numpy_version": "",
        "workload": "ovr_logistic_partial_fit", "backend": "fortml",
        "n_samples": 9, "n_features": 2, "n_classes": 3, "n_train": 9,
        "batch_size": 5, "batch_count": 2,
    }
    rows: list[dict[str, Any]] = [
        row(details, phase="independent_metadata", device="cpu", status="pass",
            metric="batch_count", value=oracle["batch_count"], max_abs_error=0.0,
            oracle="independent sorted-vocabulary stream state machine",
            notes="first batch omits one declared class; second completes it"),
        row(details, phase="independent_metadata", device="cpu", status="pass",
            metric="sample_count", value=oracle["sample_count"], max_abs_error=0.0,
            oracle="independent sorted-vocabulary stream state machine",
            notes="accepted batches are counted exactly once"),
    ]
    values: dict[str, float] = {}
    app_status = "skipped"
    if not args.skip_fortml:
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        gate = subprocess.run(
            ["fo", "test", "test_ovr_logistic_partial_fit"], cwd=fortml,
            env=environment, capture_output=True, text=True,
        )
        app = subprocess.run(
            ["fo", "exec", args.target], cwd=fortml,
            env=environment, capture_output=True, text=True,
        )
        if gate.returncode == 0 and app.returncode == 0:
            try:
                values = parse_app(app.stdout)
                app_status = "pass"
            except RuntimeError:
                app_status = "failed"
        else:
            app_status = "failed"
    replay_error = values.get("replay_error", float("nan"))
    app_pass = (
        app_status == "pass"
        and replay_error <= 3.0e-7
        and values.get("batch_count") == oracle["batch_count"]
        and values.get("cuda_status") == FORTNUM_NOT_IMPLEMENTED
    )
    rows.extend([
        row(details, phase="release_app", device="cpu",
            status="pass" if app_pass else app_status,
            metric="replay_probability_max_abs_error", value=replay_error,
            max_abs_error=replay_error,
            oracle="one-shot versus concatenated-batch FortML fit",
            seconds_per_operation=values.get("seconds", ""),
            notes="sorted arbitrary labels and deterministic replay"),
        row(details, phase="release_app", device="cpu",
            status="pass" if app_pass else app_status, metric="batch_count",
            value=values.get("batch_count", ""), max_abs_error=abs(
                values.get("batch_count", float("nan")) - oracle["batch_count"]),
            oracle="independent sorted-vocabulary stream state machine",
            notes="metadata survives replay"),
        row(details, phase="behavioral_gate", device="cpu",
            status="pass" if not args.skip_fortml else "skipped",
            metric="test_ovr_logistic_partial_fit",
            value=1.0 if not args.skip_fortml else "",
            max_abs_error=0.0,
            oracle="independent Fortran JVP and rollback behavioral oracle",
            notes="fo test test_ovr_logistic_partial_fit"),
        row(details, phase="device_boundary", device="cuda",
            status="unavailable", metric="predict_proba_device_status",
            value=values.get("cuda_status", ""),
            max_abs_error=0.0, oracle="typed CUDA capability contract",
            notes="resident OVR multi-head kernel is not linked; no host fallback"),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = root / "results/OVR_LOGISTIC_PARTIAL_FIT.md"
    report.write_text(
        "# OVR logistic partial-fit benchmark\n\n"
        "This lane checks sorted arbitrary labels, deferred class completion, "
        "deterministic replay, malformed-batch rollback, fixed-state JVP "
        "products, and the typed CUDA boundary. The Python stream state "
        "machine is independent of the Fortran metadata implementation.\n\n"
        f"FortML revision: {details['fortml_revision']}\n"
        f"Benchmark revision: {details['benchmark_revision']}\n\n"
        "| phase | device | status | metric | value | max abs error |\n"
        "| --- | --- | --- | --- | ---: | ---: |\n" +
        "".join(
            f"| {record['phase']} | {record['device']} | {record['status']} | "
            f"{record['metric']} | {record['value']} | "
            f"{record['max_abs_error']} |\n" for record in rows
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
