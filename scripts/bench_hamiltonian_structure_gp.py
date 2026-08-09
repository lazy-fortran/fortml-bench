#!/usr/bin/env python3
"""Correctness-gated separable-Hamiltonian finite-feature GP benchmark.

NumPy independently solves the two frozen-feature kernel-ridge systems for
``V(q)`` and ``T(p)``. The FortML release app is retained only when both RMSEs
match and it reports a zero structure defect. CUDA is an explicit unavailable
row because no resident structure-aware Hamiltonian GP graph exists.
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


N_SAMPLES = 256
N_COORDINATES = 2
N_HIDDEN = 16
REGULARIZATION = 0.1
REPETITIONS = 8
RMSE_TOLERANCE = 2.0e-12
STRUCTURE_TOLERANCE = 2.0e-13
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_coordinates", "n_hidden", "regularization", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error",
    "kinetic_value", "structure_defect", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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


def layer(seed: int, layer_index: int, n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(6.0 / (n_in + n_out))
    indices = np.arange(1, n_in * n_out + 1, dtype=np.float64).reshape(
        (n_in, n_out), order="F"
    )
    weights = scale * np.sin(seed + 1009 * layer_index + 9176 * indices)
    bias_indices = np.arange(1, n_out + 1, dtype=np.float64)
    biases = 0.01 * scale * np.sin(seed + 1009 * layer_index + 7919 * bias_indices)
    return weights, biases


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_COORDINATES + 1, dtype=np.float64)[None, :]
    q = np.sin(0.017 * rows + 0.053 * columns)
    p = np.cos(0.011 * rows * columns)
    potential = 0.5 * np.sum(q**2, axis=1, keepdims=True)
    kinetic = 0.5 * np.sum(p**2, axis=1, keepdims=True)
    return q, potential, p, kinetic


def posterior(
    x: np.ndarray, target: np.ndarray, seed: int
) -> tuple[np.ndarray, float, float, float]:
    weights, biases = layer(seed, 1, N_COORDINATES, N_HIDDEN)
    hidden = np.tanh(x @ weights + biases)
    design = np.concatenate([hidden, np.ones((N_SAMPLES, 1))], axis=1)
    started = time.perf_counter()
    coefficients = np.linalg.solve(
        design.T @ design + REGULARIZATION * np.eye(N_HIDDEN + 1),
        design.T @ target,
    )
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        prediction = design @ coefficients
    predict_seconds = (time.perf_counter() - started) / REPETITIONS
    rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("NumPy Hamiltonian posterior is nonfinite")
    return coefficients, fit_seconds, predict_seconds, rmse


def row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "hamiltonian_structure_gp", "phase": "predict", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_coordinates": N_COORDINATES, "n_hidden": N_HIDDEN,
        "regularization": REGULARIZATION, "repetitions": REPETITIONS,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "kinetic_value": "", "structure_defect": "",
        "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def run_fortml(
    fortml: Path, target: str, details: dict[str, Any], expected_potential: float,
    expected_kinetic: float,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml,
        env=environment, capture_output=True, text=True,
    )
    if build.returncode:
        note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
        return row(details, backend="fortml", status="unavailable",
                   oracle="FortML release-app protocol", notes=note)
    run = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml,
        env=environment, capture_output=True, text=True,
    )
    if run.returncode:
        note = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "release app failed"
        return row(details, backend="fortml", status="unavailable",
                   oracle="FortML release-app protocol", notes=note)
    pattern = re.compile(
        r"^hamiltonian_structure_gp,\s*(\d+),\s*(\d+),\s*(\d+),\s*"
        r"([0-9Ee+.-]+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+),\s*"
        r"([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$"
    )
    match = next((pattern.match(line.strip()) for line in run.stdout.splitlines()
                  if pattern.match(line.strip())), None)
    if match is None:
        return row(details, backend="fortml", status="unavailable",
                   oracle="FortML release-app protocol",
                   notes=f"timing record not found: {run.stdout!r}")
    fit_seconds, predict_seconds = float(match.group(4)), float(match.group(5))
    potential_rmse, kinetic_rmse, structure_defect = map(float, match.group(6, 7, 8))
    potential_error = abs(potential_rmse - expected_potential)
    kinetic_error = abs(kinetic_rmse - expected_kinetic)
    if potential_error > RMSE_TOLERANCE or kinetic_error > RMSE_TOLERANCE:
        raise RuntimeError(
            f"FortML Hamiltonian RMSE mismatch: V={potential_error:.3e}, T={kinetic_error:.3e}"
        )
    if structure_defect > STRUCTURE_TOLERANCE:
        raise RuntimeError(f"FortML structure-defect mismatch: {structure_defect:.3e}")
    return row(
        details, backend="fortml", status="pass", phase="predict",
        seconds_per_operation=predict_seconds, metric="potential_fit_rmse",
        value=potential_rmse, kinetic_value=kinetic_rmse,
        max_abs_error=max(potential_error, kinetic_error),
        structure_defect=structure_defect,
        oracle="independent NumPy finite-feature V/T kernel-ridge solve",
        notes=f"RMSE tolerance={RMSE_TOLERANCE:.1e}; fit_seconds={fit_seconds:.16e}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/hamiltonian_structure_gp.csv"))
    parser.add_argument("--target", default="fortml_bench_hamiltonian_structure_gp")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    q, potential_target, p, kinetic_target = fixture()
    _, potential_fit, potential_predict, potential_rmse = posterior(q, potential_target, 29)
    _, kinetic_fit, kinetic_predict, kinetic_rmse = posterior(p, kinetic_target, 29 + 7919)
    rows = [row(
        details, backend="numpy_oracle", status="pass", phase="fit",
        seconds_per_operation=potential_fit + kinetic_fit,
        metric="potential_fit_rmse", value=potential_rmse,
        kinetic_value=kinetic_rmse, max_abs_error=0.0, structure_defect=0.0,
        oracle="independent NumPy finite-feature V/T kernel-ridge solve",
        notes=f"potential_predict_seconds={potential_predict:.16e}; kinetic_predict_seconds={kinetic_predict:.16e}",
    )]
    if args.skip_fortml:
        rows.append(row(details, backend="fortml", status="skipped",
                        oracle="FortML release-app protocol", notes="--skip-fortml"))
    else:
        rows.append(run_fortml(fortml, args.target, details, potential_rmse, kinetic_rmse))
    rows.append(row(
        details, phase="device_capability", backend="fortml", device="cuda",
        status="unavailable", oracle="typed_device_contract",
        notes="resident CUDA separable Hamiltonian GP/MLP kernels are not implemented",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
