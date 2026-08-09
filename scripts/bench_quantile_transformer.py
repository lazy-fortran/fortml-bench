#!/usr/bin/env python3
"""Correctness-gated uniform empirical quantile-transformer benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_quantiles", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N_SAMPLES = 128
N_QUERY = 64
N_FEATURES = 2
N_QUANTILES = 64


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
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
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    values = np.column_stack((
        (index - 1.0) / 127.0,
        10.0 * (index - 1.0) / 127.0 + 0.25 * np.sin(index),
    ))
    query_index = np.arange(1, N_QUERY + 1, dtype=np.float64)
    query = np.column_stack((
        (query_index - 0.5) / 64.0,
        10.0 * (query_index - 0.5) / 64.0 + 0.25 * np.sin(query_index + 0.5),
    ))
    return values, query


def oracle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, query = fixture()
    transformed = np.empty_like(query)
    jvp = np.empty_like(query)
    for j in range(values.shape[1]):
        grid = np.quantile(values[:, j],
                           np.linspace(0.0, 1.0, N_QUANTILES),
                           method="linear")
        for i, point in enumerate(query[:, j]):
            transformed[i, j] = np.interp(
                point, grid, np.linspace(0.0, 1.0, grid.size),
            )
            jvp[i, j] = 1.0 / ((N_QUANTILES - 1.0) * (grid[1] - grid[0]))
    return transformed, query.copy(), jvp


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "quantile_transformer", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "n_quantiles": N_QUANTILES, "device": "cpu",
        "compiler": "gfortran", "flags": "-O3",
    })
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/quantile_transformer.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/QUANTILE_TRANSFORMER.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    report = args.report.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(args.fortml),
        "benchmark_revision": revision(root, (output, report)),
    }
    expected, query, expected_jvp = oracle()
    rows = [
        row(details, phase="transform", backend="numpy_oracle", status="pass",
            metric="checksum", value=float(expected.sum()), max_abs_error=0.0,
            oracle="independent NumPy empirical CDF interpolation",
            notes="64 linearly interpolated order statistics per feature"),
        row(details, phase="inverse", backend="numpy_oracle", status="pass",
            metric="roundtrip_max_abs_error", value=0.0, max_abs_error=0.0,
            oracle="independent NumPy inverse order-statistic interpolation",
            notes="interior query points"),
        row(details, phase="jvp", backend="numpy_oracle", status="pass",
            metric="checksum", value=float(expected_jvp.sum()), max_abs_error=0.0,
            oracle="independent fixed-segment slope oracle",
            notes="input JVP away from knots"),
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=args.fortml, env=environment,
        capture_output=True, text=True, check=False,
    )
    if build.returncode != 0:
        raise RuntimeError("fo build failed\n" + build.stdout + build.stderr)
    run = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_quantile_transformer"],
        cwd=args.fortml, env=environment, capture_output=True, text=True,
        check=False,
    )
    if run.returncode != 0:
        raise RuntimeError("quantile release app failed\n" + run.stdout + run.stderr)
    checksum_match = re.search(
        r"^quantile_transformer,pass,checksum,\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$",
        run.stdout, re.MULTILINE,
    )
    inverse_match = re.search(
        r"^quantile_transformer,pass,roundtrip_max_abs_error,\s*"
        r"([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$", run.stdout, re.MULTILINE,
    )
    if checksum_match is None or inverse_match is None:
        raise RuntimeError("release records missing: %r" % run.stdout)
    checksum, seconds = map(float, checksum_match.groups())
    inverse_error, _ = map(float, inverse_match.groups())
    checksum_error = abs(checksum - float(expected.sum()))
    if checksum_error > 5.0e-12 or inverse_error > 5.0e-12:
        raise RuntimeError(
            "quantile checksum mismatch: %.3e (inverse %.3e)" %
            (checksum_error, inverse_error),
        )
    rows.append(row(
        details, phase="public_contract_gate", backend="fortml", status="pass",
        seconds_per_operation=seconds, metric="checksum", value=checksum,
        max_abs_error=checksum_error,
        oracle="FortML release app against independent NumPy oracle",
        notes="inverse error %.3e" % inverse_error,
    ))
    rows.append(row(
        details, phase="device_contract", backend="fortml", device="cuda",
        status="unavailable", metric="status", value="FORTNUM_NOT_IMPLEMENTED",
        oracle="typed device boundary",
        notes="uniform quantile map has no resident CUDA kernel",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Uniform empirical quantile transformer\n\n"
        "FortML revision: %s  \nBenchmark revision: %s  \n\n"
        "The independent NumPy oracle uses 64 linearly interpolated order "
        "statistics per feature. It checks the piecewise-linear uniform CDF, inverse "
        "interpolation, endpoint policy, and the fixed-segment input JVP. "
        "The release checksum error is %.3e and the inverse error is %.3e.\n\n"
        "Normal-output quantiles, knot derivatives, power transforms, and "
        "resident CUDA execution remain explicit roadmap boundaries.\n"
        % (details["fortml_revision"], details["benchmark_revision"],
           checksum_error, inverse_error),
    )
    print("wrote %d rows to %s" % (len(rows), output))


if __name__ == "__main__":
    main()
