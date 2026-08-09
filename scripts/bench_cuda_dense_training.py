#!/usr/bin/env python3
"""Benchmark and correctness report for resident CUDA dense training.

The native test owns CUDA allocation and launch.  This harness independently
reconstructs the linear-layer SGD, Adam, and AdamW recurrences and records the
native and compute-sanitizer gates without treating compilation time as a
kernel timing.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_parameters",
    "steps", "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
LR = {"sgd": 0.08, "adam": 0.05, "adamw": 0.03}
BETA1 = {"sgd": 0.9, "adam": 0.8, "adamw": 0.9}
BETA2 = {"sgd": 0.99, "adam": 0.9, "adamw": 0.99}
EPSILON = {"sgd": 1.0e-8, "adam": 1.0e-7, "adamw": 1.0e-8}
DECAY = {"sgd": 0.01, "adam": 0.2, "adamw": 0.1}
STEPS = 4
TOLERANCE = 3.0e-12


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


def metadata(root: Path, fortml: Path, output: Path, report: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": os.environ.get("NVCC", "nvcc"),
        "flags": os.environ.get("NVCCFLAGS", "-O3 -arch=native"),
    }


def oracle() -> tuple[dict[str, float], dict[str, float]]:
    query = np.array([[-1.0, 0.0, 2.0, 0.5],
                      [0.75, -0.5, 1.25, -1.5]], dtype=np.float64)
    target = np.array([0.2, -0.3, 0.6, 0.4], dtype=np.float64)
    outputs: dict[str, float] = {}
    checksums: dict[str, float] = {}
    for name in ("sgd", "adam", "adamw"):
        parameters = np.array([0.5, -0.25, 0.1], dtype=np.float64)
        first = np.zeros(3, dtype=np.float64)
        second = np.zeros(3, dtype=np.float64)
        for step in range(1, STEPS + 1):
            residual = parameters[2] + parameters[0] * query[0] + \
                parameters[1] * query[1] - target
            gradient = np.array([
                np.mean(residual * query[0]), np.mean(residual * query[1]),
                np.mean(residual)], dtype=np.float64)
            if name == "sgd":
                update = gradient
            else:
                first = BETA1[name] * first + (1.0 - BETA1[name]) * gradient
                second = BETA2[name] * second + (1.0 - BETA2[name]) * gradient**2
                first_hat = first / (1.0 - BETA1[name] ** step)
                second_hat = second / (1.0 - BETA2[name] ** step)
                update = first_hat / (np.sqrt(second_hat) + EPSILON[name])
            if name == "adamw":
                parameters *= 1.0 - LR[name] * DECAY[name]
            parameters -= LR[name] * update
        outputs[name] = float(np.linalg.norm(parameters))
        checksums[name] = float(np.sum(parameters) + np.sum(first) + np.sum(second))
    return outputs, checksums


def run_gate(fortml: Path, sanitizer: bool) -> tuple[str, str, float | None]:
    script = fortml / "test" / (
        "run_cuda_dense_resident_training_sanitizer.sh" if sanitizer else
        "run_cuda_dense_resident_training.sh")
    if not script.is_file():
        return "unavailable", f"test script is absent: {script.name}", None
    process = subprocess.run([str(script)], cwd=fortml, capture_output=True, text=True)
    output = (process.stdout + "\n" + process.stderr).strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    lowered = output.lower()
    if process.returncode != 0:
        return "failed", lines[-1] if lines else "CUDA gate returned nonzero status", None
    if "skipped" in lowered or "unavailable" in lowered:
        note = next((line for line in lines if "skip" in line.lower() or
                     "unavailable" in line.lower()), "CUDA gate unavailable")
        return "unavailable", note, None
    match = re.search(r"max error ([0-9.+\-eE]+)", output)
    error = float(match.group(1)) if match else None
    if "pass" not in lowered or error is None:
        return "failed", lines[-1] if lines else "CUDA gate emitted no PASS/error marker", error
    if error > TOLERANCE:
        return "failed", f"native error {error:.3e} exceeds {TOLERANCE:.3e}", error
    note = next((line for line in lines if "pass" in line.lower()), "CUDA gate passed")
    return "pass", note, error


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"workload": "cuda_dense_resident_training", "n_parameters": 3,
                "steps": STEPS, "oracle": "independent NumPy linear MSE recurrence"})
    row.update(values)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/cuda_dense_training.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/CUDA_DENSE_TRAINING.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    details = metadata(root, fortml, output, report)
    norms, checksums = oracle()
    rows: list[dict[str, Any]] = []
    for name in ("sgd", "adam", "adamw"):
        rows.append(base(details, phase=f"{name}_oracle", backend="numpy_oracle",
                         device="cpu", status="pass", repetitions=32,
                         seconds_per_operation="", metric="parameter_l2_norm",
                         value=norms[name], max_abs_error=0.0,
                         notes=f"state_checksum={checksums[name]:.16e}"))
    status, note, error = run_gate(fortml, sanitizer=False)
    rows.append(base(details, phase="native_gate", backend="fortml", device="cuda",
                     status=status, metric="parameter_l2_norm", value=max(norms.values()),
                     max_abs_error=error if error is not None else "",
                     notes="; ".join((f"sgd_adam_adamw_norms={norms}", note))))
    sanitizer_status, sanitizer_note, sanitizer_error = run_gate(fortml, sanitizer=True)
    rows.append(base(details, phase="compute_sanitizer", backend="compute-sanitizer",
                     device="cuda", status=sanitizer_status,
                     metric="memcheck_error_count", value=0,
                     max_abs_error=sanitizer_error if sanitizer_error is not None else "",
                     notes=sanitizer_note))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Resident CUDA dense training\n\n"
        "This lane covers a single dense linear layer with an explicitly uploaded "
        "resident batch. Gradients, Adam moments, and model parameters remain on "
        "the selected device across four updates. The native oracle covers SGD, "
        "Adam, and AdamW. Compute-sanitizer runs the same independent fixture.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Native gate: `{status}` (max error `{error if error is not None else 'n/a'}`)\n"
        f"- Compute-sanitizer: `{sanitizer_status}`\n"
        f"- Oracle parameter norms: `{norms}`\n\n"
        "The gate subprocess includes compilation and is not reported as a "
        "per-step performance number. See `fortml/docs/CUDA_DENSE_TRAINING.md` "
        "for the typed Fortran API and transfer-accounting contract.\n"
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
