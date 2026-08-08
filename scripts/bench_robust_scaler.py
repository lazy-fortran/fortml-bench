#!/usr/bin/env python3
"""Correctness-gated benchmark for dense median/IQR scaling.

The NumPy path is an independent reference implementation of the release
fixture.  FortML's release app reports transform and JVP checksums; those
checksums are compared before CPU timing is retained.  CUDA is represented by
an explicit unavailable row until a resident preprocessing kernel exists.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES, N_FEATURES = 96, 5
REPETITIONS = 256
CHECKSUM_TOLERANCE = 5.0e-11
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return HEAD plus a dirty marker, ignoring generated result files."""

    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        path = (repository / name).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def fixture() -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the release app's deterministic 96-by-5 fixture."""

    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    values = np.sin(0.031 * rows + 0.17 * columns) + 0.01 * np.mod(rows * columns, 13.0)
    tangents = np.cos(0.023 * (rows + 2.0 * columns))
    return values, tangents


def robust_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent linear-interpolation median/IQR reference."""

    center = np.percentile(values, 50.0, axis=0, method="linear")
    lower = np.percentile(values, 25.0, axis=0, method="linear")
    upper = np.percentile(values, 75.0, axis=0, method="linear")
    scale = np.where(upper > lower, upper - lower, 1.0)
    return center, scale


def base_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict(details)
    row.update({
        "workload": "robust_scaler", "phase": "", "backend": "", "device": "cpu",
        "status": "", "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "repetitions": "", "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def numpy_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    values, tangents = fixture()
    center, scale = robust_fit(values)
    transformed = (values - center[None, :]) / scale[None, :]
    recovered = transformed * scale[None, :] + center[None, :]
    transformed_tangent = tangents / scale[None, :]
    rows: list[dict[str, Any]] = []

    def timed(operation: Any) -> float:
        started = time.perf_counter()
        for _ in range(REPETITIONS):
            operation()
        return (time.perf_counter() - started) / REPETITIONS

    fit_seconds = timed(lambda: robust_fit(values))
    transform_seconds = timed(lambda: (values - center[None, :]) / scale[None, :])
    inverse_seconds = timed(lambda: transformed * scale[None, :] + center[None, :])
    jvp_seconds = timed(lambda: tangents / scale[None, :])
    oracle = "independent NumPy linear-interpolation median/IQR oracle"
    rows.extend([
        base_row(details, phase="fit", backend="numpy_oracle", status="pass",
                 repetitions=REPETITIONS, seconds_per_operation=fit_seconds,
                 metric="center_scale_checksum", value=float(center.sum() + scale.sum()),
                 max_abs_error=0.0, oracle=oracle,
                 notes=f"center checksum={float(center.sum()):.17g}; scale checksum={float(scale.sum()):.17g}"),
        base_row(details, phase="transform", backend="numpy_oracle", status="pass",
                 repetitions=REPETITIONS, seconds_per_operation=transform_seconds,
                 metric="transformed_checksum", value=float(transformed.sum()),
                 max_abs_error=0.0, oracle=oracle, notes="complete dense transform array"),
        base_row(details, phase="inverse", backend="numpy_oracle", status="pass",
                 repetitions=REPETITIONS, seconds_per_operation=inverse_seconds,
                 metric="inverse_checksum", value=float(recovered.sum()), max_abs_error=0.0,
                 oracle="independent NumPy inverse affine oracle",
                 notes=f"max reconstruction error={float(np.max(np.abs(recovered - values))):.3e}"),
        base_row(details, phase="jvp", backend="numpy_oracle", status="pass",
                 repetitions=REPETITIONS, seconds_per_operation=jvp_seconds,
                 metric="jvp_checksum", value=float(transformed_tangent.sum()),
                 max_abs_error=0.0, oracle="independent NumPy diagonal affine JVP oracle",
                 notes="input tangent divided by fitted IQR"),
    ])
    return rows


def run_fortml(fortml: Path, details: dict[str, str], no_build: bool,
               expected_transform: float, expected_jvp: float) -> list[dict[str, Any]]:
    target = "fortml_bench_robust_scaler"
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return [base_row(details, phase="transform", backend="fortml_cpu", status="unavailable",
                         oracle="FortML release-app protocol",
                         notes=f"release target source is absent: {source.name}")]
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    if not no_build:
        build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                               env=environment, capture_output=True, text=True)
        if build.returncode != 0:
            note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
            return [base_row(details, phase="transform", backend="fortml_cpu", status="unavailable",
                             oracle="FortML release-app protocol", notes=note)]
    completed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=environment, capture_output=True, text=True)
    if completed.returncode != 0:
        note = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "release app failed"
        return [base_row(details, phase="transform", backend="fortml_cpu", status="unavailable",
                         oracle="FortML release-app protocol", notes=note)]
    match = re.search(
        r"^robust_scaler,\s*96,\s*5,\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)\s*$",
        completed.stdout, re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"FortML robust-scaler record not found: {completed.stdout!r}")
    elapsed, transformed_checksum, jvp_checksum = map(float, match.groups())
    transform_error = abs(transformed_checksum - expected_transform)
    jvp_error = abs(jvp_checksum - expected_jvp)
    if max(transform_error, jvp_error) > CHECKSUM_TOLERANCE:
        raise RuntimeError(
            "FortML robust-scaler checksum mismatch: "
            f"transform={transform_error:.3e}, jvp={jvp_error:.3e}"
        )
    oracle = "independent NumPy median/IQR transform and JVP checksum oracle"
    return [
        base_row(details, phase="transform", backend="fortml_cpu", status="pass",
                 repetitions=REPETITIONS, seconds_per_operation=elapsed,
                 metric="transformed_checksum", value=transformed_checksum,
                 max_abs_error=transform_error, oracle=oracle,
                 notes="fortml_bench_robust_scaler release timing"),
        base_row(details, phase="jvp", backend="fortml_cpu", status="pass",
                 repetitions=REPETITIONS, metric="jvp_checksum", value=jvp_checksum,
                 max_abs_error=jvp_error, oracle=oracle,
                 notes="checksum gate passed; release app does not emit a separate JVP timing"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/robust_scaler.csv"))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, output)
    values, _ = fixture()
    center, scale = robust_fit(values)
    expected_transform = float(((values - center[None, :]) / scale[None, :]).sum())
    expected_jvp = float((fixture()[1] / scale[None, :]).sum())
    rows = numpy_rows(details)
    rows.extend(run_fortml(fortml, details, args.no_build, expected_transform, expected_jvp))
    rows.append(base_row(details, phase="transform", backend="fortml_cuda", device="cuda",
                         status="unavailable", metric="transformed_checksum",
                         oracle="typed device contract",
                         notes="resident robust-scaler kernel is not linked; no host fallback"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
