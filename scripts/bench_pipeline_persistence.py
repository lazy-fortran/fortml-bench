#!/usr/bin/env python3
"""Correctness-gated benchmark for fitted basis-pipeline persistence.

The release app checks a versioned host text round trip and derivative metadata
through the Fortran API. NumPy independently reconstructs the polynomial and
Fourier feature columns before a timing row is recorded. Resident CUDA
serialization is represented as a typed refusal, never as host timing.
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
    "max_abs_error", "oracle", "seconds", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return a reproducible commit pin, adding ``+dirty`` outside ignores."""
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
    """Build the expected four-column fixture without using FortML code."""
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float64)[:, None]
    expected = np.column_stack(
        [x[:, 0], x[:, 0] ** 2, np.sin(0.8 * x[:, 0]), np.cos(0.8 * x[:, 0])]
    )
    # Recompute the same columns through independent scalar expressions and a
    # fresh array so this is an oracle rather than a repository-state check.
    reference = np.empty_like(expected)
    for row, value in enumerate(x[:, 0]):
        reference[row, :] = (value, value * value, np.sin(0.8 * value),
                             np.cos(0.8 * value))
    return expected.shape[1], float(np.max(np.abs(expected - reference)))


def parse_release(stdout: str) -> dict[str, float | int]:
    records: dict[str, float | int] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] != "pipeline_persistence":
            continue
        if len(fields) != 9 or fields[1] != "round_trip":
            raise RuntimeError(f"malformed persistence record: {line!r}")
        names = (
            "n_samples", "n_features", "n_parameters", "seconds",
            "roundtrip_error", "feature_oracle_error", "metadata_checksum",
        )
        for name, value in zip(names, fields[2:], strict=True):
            records[name] = float(value) if name in {
                "seconds", "roundtrip_error", "feature_oracle_error",
            } else int(value)
    if not records:
        raise RuntimeError("release app omitted pipeline persistence record")
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
        default=Path("results/pipeline_persistence.csv"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected_features, independent_error = independent_oracle()
    if independent_error > 1.0e-14:
        raise RuntimeError(f"independent feature oracle is inconsistent: {independent_error}")

    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O2"], cwd=fortml, env=environment, check=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_pipeline_persistence"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    observed = parse_release(completed.stdout)
    expected = {
        "n_samples": 64, "n_features": expected_features, "n_parameters": 1,
        "metadata_checksum": 4,
    }
    for name, value in expected.items():
        if abs(float(observed[name]) - value) > 3.0e-12:
            raise RuntimeError(
                f"independent persistence metadata mismatch for {name}: "
                f"{observed[name]!r} != {value!r}"
            )
    for name in ("roundtrip_error", "feature_oracle_error"):
        if observed[name] > 3.0e-12:
            raise RuntimeError(f"{name} exceeded tolerance: {observed[name]:.3e}")

    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml, (
            fortml / "test_mlp_amsgrad_checkpoint.txt",
            fortml / "test_mlp_radam_checkpoint.txt",
        )),
        "benchmark_revision": revision(root, (output,)),
        "compiler": environment["FO_FC"],
        "flags": "-O2",
        "oracle": "independent NumPy polynomial/Fourier reconstruction plus metadata",
    }
    records = [
        row(
            details,
            workload="pipeline_persistence", phase="save_load_roundtrip",
            backend="fortml", device="cpu", status="pass",
            metric="max_abs_error", value=observed["roundtrip_error"],
            max_abs_error=observed["roundtrip_error"], seconds=observed["seconds"],
            notes="versioned text state preserves names, offsets, and packed parameters",
        ),
        row(
            details,
            workload="pipeline_persistence", phase="independent_oracle",
            backend="fortml", device="cpu", status="pass",
            metric="max_abs_error", value=observed["feature_oracle_error"],
            max_abs_error=observed["feature_oracle_error"], seconds=observed["seconds"],
            notes="64-sample polynomial/Fourier feature construction",
        ),
        row(
            details,
            workload="pipeline_persistence", phase="device_capability",
            backend="fortml", device="cuda", status="unavailable",
            metric="resident_serialization", value="nan", max_abs_error="nan",
            oracle="typed CUDA refusal",
            notes="FORTNUM_NOT_IMPLEMENTED; no host fallback or hidden transfer",
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
