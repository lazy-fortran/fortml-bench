#!/usr/bin/env python3
"""Correctness-gated OVR XGBoost multiclass text persistence benchmark.

The release app fits three arbitrary integer classes, checks the independent
stable-sigmoid probability reconstruction, and round-trips the complete OVR
ensemble through one text artifact.  NumPy independently checks the simplex
and label metadata before any timing is retained.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "seconds", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(
                path.resolve().relative_to(repository.resolve()).as_posix()
            )
        except ValueError:
            continue
    dirty = any(
        line[3:].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def parse_release(stdout: str) -> dict[str, float | int]:
    records: dict[str, float | int] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] != "xgb_multiclass_persistence":
            continue
        if len(fields) != 11 or fields[1] != "round_trip":
            raise RuntimeError(f"malformed persistence record: {line!r}")
        names = (
            "n_samples", "n_features", "n_classes", "n_estimators", "seconds",
            "roundtrip_error", "oracle_error", "probability_sum", "class_checksum",
        )
        for name, value in zip(names, fields[2:], strict=True):
            records[name] = float(value) if name in {
                "seconds", "roundtrip_error", "oracle_error", "probability_sum",
            } else int(value)
    if not records:
        raise RuntimeError("release app omitted multiclass persistence record")
    return records


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/xgboost_multiclass_persistence.csv"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O2"], cwd=fortml, env=environment, check=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_xgboost_multiclass_persistence"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    observed = parse_release(completed.stdout)
    expected = {
        "n_samples": 9, "n_features": 1, "n_classes": 3, "n_estimators": 3,
        "probability_sum": 3.0, "class_checksum": -8 + 2 + 11,
    }
    for name, value in expected.items():
        if abs(float(observed[name]) - value) > 3.0e-12:
            raise RuntimeError(
                f"independent multiclass metadata oracle mismatch for {name}: "
                f"{observed[name]!r} != {value!r}"
            )
    if observed["roundtrip_error"] > 3.0e-12:
        raise RuntimeError(
            f"multiclass text round-trip mismatch: {observed['roundtrip_error']:.3e}"
        )
    if observed["oracle_error"] > 3.0e-12:
        raise RuntimeError(
            f"independent multiclass probability mismatch: {observed['oracle_error']:.3e}"
        )

    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": "not used",
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O2",
        "oracle": "independent probability-simplex and arbitrary-label oracle",
    }
    records = [
        row(
            details,
            workload="xgboost_multiclass_persistence",
            phase="save_load_roundtrip", backend="fortml", device="cpu",
            status="pass", metric="max_abs_error",
            value=observed["roundtrip_error"],
            max_abs_error=observed["roundtrip_error"], seconds=observed["seconds"],
            notes="one-file OVR snapshot preserves all class probabilities",
        ),
        row(
            details,
            workload="xgboost_multiclass_persistence",
            phase="independent_oracle", backend="fortml", device="cpu",
            status="pass", metric="max_abs_error", value=observed["oracle_error"],
            max_abs_error=observed["oracle_error"], seconds=observed["seconds"],
            notes="stable sigmoid margins, simplex sum=3, labels=[-8,2,11]",
        ),
        row(
            details,
            workload="xgboost_multiclass_persistence",
            phase="device_capability", backend="fortml", device="cuda",
            status="unavailable", metric="resident_tree_persistence", value="nan",
            max_abs_error="nan", oracle="typed device contract",
            notes="no resident CUDA tree kernel; persistence is CPU text I/O",
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
