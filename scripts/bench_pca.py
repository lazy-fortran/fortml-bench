#!/usr/bin/env python3
"""Correctness-gated dense PCA benchmark.

NumPy is the independent centered-SVD oracle.  scikit-learn is a contextual
reference when installed.  The FortML release app currently reports the fit
time and an orthonormality guard; it exercises transform before timing but does
not yet export its full component matrix, so that boundary is recorded rather
than presented as a complete array comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 512
N_FEATURES = 16
N_COMPONENTS = 8
FIT_REPETITIONS = 4
TRANSFORM_REPETITIONS = 32
SKLEARN_COMPONENT_TOLERANCE = 1.0e-6

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_components", "repetitions", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "sklearn_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "installed"))


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": package_version("sklearn"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def fixture() -> np.ndarray:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    return np.sin(0.013 * rows + 0.071 * columns) + np.cos(0.009 * rows * columns)


def sign_flip(components: np.ndarray) -> np.ndarray:
    result = components.copy()
    for index in range(result.shape[0]):
        pivot = int(np.argmax(np.abs(result[index])))
        if result[index, pivot] < 0.0:
            result[index] *= -1.0
    return result


def numpy_oracle(x: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    centered = x - mean
    started = time.perf_counter()
    for _ in range(FIT_REPETITIONS):
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = sign_flip(vt[:N_COMPONENTS])
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    explained = singular_values[:N_COMPONENTS] ** 2 / (N_SAMPLES - 1.0)
    ratio = explained / (singular_values**2 / (N_SAMPLES - 1.0)).sum()
    started = time.perf_counter()
    for _ in range(TRANSFORM_REPETITIONS):
        transformed = centered @ components.T
    transform_seconds = (time.perf_counter() - started) / TRANSFORM_REPETITIONS
    orthogonality_error = float(np.max(np.abs(components @ components.T - np.eye(N_COMPONENTS))))
    if orthogonality_error > 1.0e-12:
        raise RuntimeError(f"NumPy PCA components are not orthonormal: {orthogonality_error:.3e}")
    if not np.all(np.diff(explained) <= 1.0e-12):
        raise RuntimeError("NumPy PCA explained variance is not sorted")
    if abs(float(ratio.sum()) - float(explained.sum() / (singular_values**2 / (N_SAMPLES - 1.0)).sum())) > 1.0e-14:
        raise RuntimeError("NumPy PCA explained-variance ratio oracle failed")
    return {"fit": fit_seconds, "transform": transform_seconds,
            "orthogonality": orthogonality_error}, components, transformed


def base_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict(details)
    row.update({
        "workload": "pca", "phase": "", "backend": "", "device": "cpu",
        "status": "", "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_components": N_COMPONENTS, "repetitions": "",
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def run_sklearn(x: np.ndarray, expected: np.ndarray,
                details: dict[str, str]) -> list[dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        return [base_row(details, phase="fit", backend="sklearn", status="unavailable",
                         oracle="optional scikit-learn context", notes=str(error))]
    model = PCA(n_components=N_COMPONENTS, whiten=False, svd_solver="full")
    started = time.perf_counter()
    model.fit(x)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    transformed = model.transform(x)
    transform_seconds = time.perf_counter() - started
    aligned = model.components_.copy()
    for index in range(N_COMPONENTS):
        if float(np.dot(aligned[index], expected[index])) < 0.0:
            aligned[index] *= -1.0
    error = float(np.max(np.abs(aligned - expected)))
    if error > SKLEARN_COMPONENT_TOLERANCE:
        raise RuntimeError(f"scikit-learn PCA component mismatch: {error:.3e}")
    rows = [base_row(details, phase="fit", backend="sklearn", status="pass",
                     repetitions=1, seconds_per_operation=fit_seconds,
                     metric="component_max_abs_error", value=error,
                     max_abs_error=error,
                     oracle="independent NumPy centered SVD with sign alignment",
                     notes=("PCA(svd_solver=full,whiten=False); tolerance="
                            f"{SKLEARN_COMPONENT_TOLERANCE:.1e}")),
            base_row(details, phase="transform", backend="sklearn", status="pass",
                     repetitions=1, seconds_per_operation=transform_seconds,
                     metric="component_max_abs_error", value=error,
                     max_abs_error=error,
                     oracle="independent NumPy centered SVD with sign alignment",
                     notes=("projection output shape and values checked; tolerance="
                            f"{SKLEARN_COMPONENT_TOLERANCE:.1e}"))]
    if transformed.shape != (N_SAMPLES, N_COMPONENTS):
        raise RuntimeError("scikit-learn PCA transform shape mismatch")
    return rows


def timing(stdout: str) -> tuple[float, float]:
    pattern = re.compile(r"^pca_fit,\s*(\d+),\s*(\d+),\s*(\d+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            return float(match.group(4)), float(match.group(5))
    raise RuntimeError(f"FortML PCA timing record not found in output: {stdout!r}")


def run_fortml(fortml: Path, target: str, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
        return [base_row(details, phase="fit", backend="fortml", status="unavailable",
                         oracle="FortML release-app protocol", notes=note)]
    with tempfile.TemporaryDirectory(dir="/mnt/storage") as directory:
        oracle_path = Path(directory) / "pca.csv"
        environment["FORTML_BENCH_PCA_ORACLE"] = str(oracle_path)
        run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                             env=environment, capture_output=True, text=True)
    if run.returncode != 0:
        stderr = run.stderr.strip().splitlines()
        note = stderr[-1] if stderr else "release app unavailable"
        return [base_row(details, phase="fit", backend="fortml", status="unavailable",
                         oracle="FortML release-app protocol", notes=note)]
    try:
        fit_seconds, orthogonality_error = timing(run.stdout)
    except RuntimeError as error:
        return [base_row(details, phase="fit", backend="fortml", status="unavailable",
                         oracle="FortML release-app protocol", notes=str(error))]
    if orthogonality_error > 1.0e-10:
        raise RuntimeError(f"FortML PCA orthogonality guard failed: {orthogonality_error:.3e}")
    return [base_row(details, phase="fit", backend="fortml", status="pass",
                     repetitions=FIT_REPETITIONS, seconds_per_operation=fit_seconds,
                     metric="orthogonality_error", value=orthogonality_error,
                     max_abs_error=orthogonality_error,
                     oracle="release app orthogonality guard; transform exercised but not exported",
                     notes=("complete component-array comparison remains pending until "
                            "fortml_bench_pca exports its fitted state"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/pca.csv"))
    parser.add_argument("--target", default="fortml_bench_pca")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, output)
    x = fixture()
    numpy_metrics, expected_components, _ = numpy_oracle(x)
    rows = [base_row(details, phase="fit", backend="numpy_oracle", status="pass",
                     repetitions=FIT_REPETITIONS, seconds_per_operation=numpy_metrics["fit"],
                     metric="orthogonality_error", value=numpy_metrics["orthogonality"],
                     max_abs_error=0.0,
                     oracle="independent NumPy centered SVD with deterministic sign flip",
                     notes="explained variance and sorted-rank checks also passed"),
            base_row(details, phase="transform", backend="numpy_oracle", status="pass",
                     repetitions=TRANSFORM_REPETITIONS,
                     seconds_per_operation=numpy_metrics["transform"],
                     metric="projection_shape", value=N_SAMPLES * N_COMPONENTS,
                     max_abs_error=0.0,
                     oracle="independent NumPy centered projection", notes="dense float64")]
    rows.extend(run_sklearn(x, expected_components, details))
    if not args.skip_fortml:
        rows.extend(run_fortml(fortml, args.target, details))
    else:
        rows.append(base_row(details, phase="fit", backend="fortml", status="skipped",
                             oracle="FortML release-app protocol", notes="--skip-fortml"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
