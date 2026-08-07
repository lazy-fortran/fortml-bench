#!/usr/bin/env python3
"""Correctness-gated benchmark for FortML's named MLP parameter layout.

The NumPy oracle constructs the dense layer tree independently.  A tiny probe
is compiled and linked against the already-built FortML archive, so the CSV
contains the metadata returned by ``mlp_t%parameter_layout`` rather than a
copy of the expected values.  Metadata is a host-side contract today; CUDA is
therefore recorded as a typed unavailable row and is never inferred from the
CPU probe.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


LAYERS = np.asarray((3, 4, 2), dtype=np.int64)
FIELDS = (
    "workload", "phase", "backend", "device", "status", "block_index",
    "block_name", "kind", "first", "last", "rows", "columns", "trainable",
    "is_buffer", "parameter_count", "block_count", "seconds", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return a commit plus a dirty marker, excluding the output being written."""

    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        if not line:
            continue
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def structural_oracle() -> list[dict[str, Any]]:
    """Construct expected named blocks with NumPy's shape arithmetic."""

    expected: list[dict[str, Any]] = []
    first = 1
    for layer, (n_in, n_out) in enumerate(zip(LAYERS[:-1], LAYERS[1:]), 1):
        n_weight = int(np.prod((n_in, n_out), dtype=np.int64))
        expected.append({
            "block_index": layer * 2 - 1,
            "block_name": f"layer_{layer}.weight",
            "kind": "weight",
            "first": first,
            "last": first + n_weight - 1,
            "rows": int(n_in),
            "columns": int(n_out),
            "trainable": True,
            "is_buffer": False,
        })
        first += n_weight
        n_bias = int(n_out)
        expected.append({
            "block_index": layer * 2,
            "block_name": f"layer_{layer}.bias",
            "kind": "bias",
            "first": first,
            "last": first + n_bias - 1,
            "rows": int(n_out),
            "columns": 1,
            "trainable": True,
            "is_buffer": False,
        })
        first += n_bias
    return expected


def _logical(value: str) -> bool:
    if value == "T":
        return True
    if value == "F":
        return False
    raise RuntimeError(f"invalid Fortran logical in probe output: {value!r}")


def parse_probe(stdout: str) -> tuple[int, int, list[dict[str, Any]], tuple[int, int, bool]]:
    parameter_count: int | None = None
    block_count: int | None = None
    layout: list[dict[str, Any]] = []
    lookup: tuple[int, int, bool] | None = None
    for raw in stdout.splitlines():
        fields = raw.strip().split(",")
        if fields[0] == "parameter_count" and len(fields) == 2:
            parameter_count = int(fields[1])
        elif fields[0] == "parameter_block_count" and len(fields) == 2:
            block_count = int(fields[1])
        elif fields[0] == "layout" and len(fields) == 10:
            layout.append({
                "block_index": int(fields[1]),
                "block_name": fields[2],
                "kind": fields[3],
                "first": int(fields[4]),
                "last": int(fields[5]),
                "rows": int(fields[6]),
                "columns": int(fields[7]),
                "trainable": _logical(fields[8]),
                "is_buffer": _logical(fields[9]),
            })
        elif fields[0] == "range" and len(fields) == 5:
            lookup = (int(fields[2]), int(fields[3]), _logical(fields[4]))
    if parameter_count is None or block_count is None or lookup is None:
        raise RuntimeError(f"incomplete MLP layout probe output:\n{stdout}")
    return parameter_count, block_count, layout, lookup


def verify(parameter_count: int, block_count: int,
           actual: list[dict[str, Any]], lookup: tuple[int, int, bool],
           expected: list[dict[str, Any]]) -> None:
    expected_count = int(sum((block["last"] - block["first"] + 1) for block in expected))
    if parameter_count != expected_count:
        raise RuntimeError(f"parameter count mismatch: {parameter_count} != {expected_count}")
    if block_count != len(expected) or len(actual) != len(expected):
        raise RuntimeError(f"block count mismatch: {block_count}, {len(actual)} != {len(expected)}")
    for got, want in zip(actual, expected):
        if got != want:
            raise RuntimeError(f"MLP metadata mismatch:\n  got={got}\n  want={want}")
    ranges = [(block["first"], block["last"]) for block in actual]
    if ranges != [(block["first"], block["last"]) for block in expected]:
        raise RuntimeError("parameter ranges are not deterministic and contiguous")
    if sum(last - first + 1 for first, last in ranges) != parameter_count:
        raise RuntimeError("parameter ranges do not cover the packed vector")
    if lookup != (17, 24, True):
        raise RuntimeError(f"named range lookup mismatch: {lookup}")


def build_probe(fortml: Path, probe_source: Path, executable: Path) -> tuple[str, float]:
    """Build FortML once, then link the probe to the newest package archive."""

    build = subprocess.run(
        ["fo", "build", "--flag", "-O2"], cwd=fortml,
        capture_output=True, text=True, check=False,
    )
    if build.returncode:
        raise RuntimeError(build.stderr.strip() or build.stdout.strip())
    library_dir = fortml / "build" / "fo" / "lib"
    archives = list(library_dir.glob("*.a"))
    if not archives:
        raise RuntimeError(f"FortML build produced no archives in {library_dir}")
    archive = max(archives, key=lambda path: path.stat().st_mtime_ns)
    module_dir = fortml / "build" / "fo" / "mod"
    compiler = shlex.split(os.environ.get("FO_FC", "gfortran"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError(f"Fortran compiler is unavailable: {compiler!r}")
    command = compiler + [
        "-O2", "-ffree-line-length-none", "-I", str(module_dir),
        str(probe_source), str(archive), "-o", str(executable),
    ]
    link = subprocess.run(command, cwd=fortml, capture_output=True, text=True, check=False)
    if link.returncode:
        raise RuntimeError(link.stderr.strip() or link.stdout.strip())
    started = time.perf_counter()
    run = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(run.stderr.strip() or run.stdout.strip())
    return run.stdout, elapsed


def csv_row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update(values)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_parameter_layout.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected = structural_oracle()
    fixture = root / "fixtures" / "mlp_parameter_layout_probe.f90"
    if not fixture.is_file():
        raise RuntimeError(f"missing probe fixture: {fixture}")
    with tempfile.TemporaryDirectory(prefix="fortml-mlp-layout-", dir="/mnt/storage") as directory:
        directory_path = Path(directory)
        source = directory_path / fixture.name
        executable = directory_path / "mlp_parameter_layout_probe"
        source.write_bytes(fixture.read_bytes())
        stdout, elapsed = build_probe(fortml, source, executable)
    parameter_count, block_count, actual, lookup = parse_probe(stdout)
    verify(parameter_count, block_count, actual, lookup, expected)
    source_revision = revision(fortml)
    benchmark_revision = revision(root, (args.output.resolve(),))
    compiler = os.environ.get("FO_FC", "gfortran")
    details = {
        "workload": "mlp_parameter_layout",
        "status": "pass",
        "oracle": "independent NumPy dense-layer shape/range oracle",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": source_revision,
        "benchmark_revision": benchmark_revision,
        "compiler": compiler,
        "flags": "-O2",
        "parameter_count": parameter_count,
        "block_count": block_count,
        "seconds": f"{elapsed:.17e}",
    }
    rows: list[dict[str, Any]] = []
    rows.append(csv_row(details, phase="parameter_count", backend="fortml", device="cpu",
                        parameter_count=parameter_count, block_count=block_count,
                        max_abs_error=0.0, notes="runtime metadata probe"))
    for block in actual:
        rows.append(csv_row(details, phase="block", backend="fortml", device="cpu",
                            **block, max_abs_error=0.0, notes="runtime metadata probe"))
    rows.append(csv_row(details, phase="range_lookup", backend="fortml", device="cpu",
                        block_name="layer_2.weight", first=lookup[0], last=lookup[1],
                        max_abs_error=0.0, notes="parameter_range lookup"))
    rows.append(csv_row(details, phase="parameter_count", backend="numpy_oracle", device="cpu",
                        parameter_count=int(sum(block["last"] - block["first"] + 1 for block in expected)),
                        block_count=len(expected), max_abs_error=0.0,
                        notes="independent structural oracle"))
    for block in expected:
        rows.append(csv_row(details, phase="block", backend="numpy_oracle", device="cpu",
                            **block, max_abs_error=0.0, notes="independent structural oracle"))
    rows.append(csv_row(details, phase="range_lookup", backend="numpy_oracle", device="cpu",
                        block_name="layer_2.weight", first=17, last=24,
                        max_abs_error=0.0, notes="independent structural oracle"))
    rows.append(csv_row(
        details, phase="device_capability", backend="fortml", device="cuda",
        status="unavailable", oracle="typed_device_contract", seconds="",
        max_abs_error="", notes="no resident CUDA metadata path; CPU data is not relabeled",
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
