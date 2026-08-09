#!/usr/bin/env python3
"""Correctness-gated benchmark for interval-routed basis feature unions.

The NumPy fixture independently assembles two selected-column Fourier branches,
checks value/JVP/VJP/HVP identities away from route endpoints, and then runs
the FortML focused oracle plus release workload. CUDA is recorded as a typed
unavailable contract; no host fallback is accepted.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_branches", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N_SAMPLES = 2048
N_FEATURES = 4


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(N_SAMPLES, dtype=np.float64)
    route = -1.0 + 2.0 * (np.mod(indices, 200.0) + 0.5) / 200.0
    signal = np.sin(0.017 * (indices + 1.0))
    x = np.column_stack((route, signal))
    x_dot = np.column_stack((np.zeros_like(route), np.cos(0.017 * (indices + 1.0))))
    theta = np.log(np.array([0.8, 0.8], dtype=np.float64))
    theta_dot = np.array([0.17, -0.23], dtype=np.float64)
    return x, x_dot, theta, theta_dot


def evaluate(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    result = np.zeros((x.shape[0], N_FEATURES), dtype=np.float64)
    for branch, mask in enumerate((x[:, 0] < 0.0, x[:, 0] >= 0.0)):
        argument = np.exp(theta[branch]) * x[:, 1]
        result[mask, 2 * branch] = np.sin(argument[mask])
        result[mask, 2 * branch + 1] = np.cos(argument[mask])
    return result


def independent_oracle() -> tuple[float, float, float, float, float]:
    x, x_dot, theta, theta_dot = fixture()
    value = evaluate(x, theta)
    h = 2.0e-6
    finite_difference = (evaluate(x + h * x_dot, theta + h * theta_dot) -
                         evaluate(x - h * x_dot, theta - h * theta_dot)) / (2.0 * h)
    tangent = np.zeros_like(value)
    theta_bar = np.zeros(2, dtype=np.float64)
    x_bar = np.zeros_like(x)
    u = (0.03 * np.arange(value.size, dtype=np.float64) - 0.17).reshape(value.shape)
    for branch, mask in enumerate((x[:, 0] < 0.0, x[:, 0] >= 0.0)):
        frequency = np.exp(theta[branch])
        argument = frequency * x[:, 1]
        argument_dot = frequency * (x_dot[:, 1] + x[:, 1] * theta_dot[branch])
        tangent[mask, 2 * branch] = np.cos(argument[mask]) * argument_dot[mask]
        tangent[mask, 2 * branch + 1] = -np.sin(argument[mask]) * argument_dot[mask]
        q = (np.cos(argument) * u[:, 2 * branch] -
             np.sin(argument) * u[:, 2 * branch + 1])
        theta_bar[branch] = np.sum(frequency * x[:, 1][mask] * q[mask])
        x_bar[mask, 1] += frequency * q[mask]
    adjoint_error = abs(np.sum(u * tangent) -
                        (np.dot(theta_bar, theta_dot) + np.sum(x_bar * x_dot)))

    def vjp(xv: np.ndarray, tv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bars = np.zeros(2, dtype=np.float64)
        xbars = np.zeros_like(xv)
        for branch, mask in enumerate((xv[:, 0] < 0.0, xv[:, 0] >= 0.0)):
            frequency = np.exp(tv[branch])
            argument = frequency * xv[:, 1]
            q = (np.cos(argument) * u[:, 2 * branch] -
                 np.sin(argument) * u[:, 2 * branch + 1])
            bars[branch] = np.sum(frequency * xv[:, 1][mask] * q[mask])
            xbars[mask, 1] += frequency * q[mask]
        return bars, xbars

    theta_plus, x_bar_plus = vjp(x + h * x_dot, theta + h * theta_dot)
    theta_minus, x_bar_minus = vjp(x - h * x_dot, theta - h * theta_dot)
    theta_hvp = np.zeros(2, dtype=np.float64)
    x_hvp = np.zeros_like(x)
    for branch, mask in enumerate((x[:, 0] < 0.0, x[:, 0] >= 0.0)):
        frequency = np.exp(theta[branch])
        argument = frequency * x[:, 1]
        argument_dot = frequency * (x_dot[:, 1] + x[:, 1] * theta_dot[branch])
        q = (np.cos(argument) * u[:, 2 * branch] -
             np.sin(argument) * u[:, 2 * branch + 1])
        q_dot = -argument_dot * (np.sin(argument) * u[:, 2 * branch] +
                                 np.cos(argument) * u[:, 2 * branch + 1])
        theta_hvp[branch] = np.sum(((frequency * theta_dot[branch] * x[:, 1] +
                                     frequency * x_dot[:, 1]) * q +
                                    frequency * x[:, 1] * q_dot)[mask])
        x_hvp[mask, 1] = frequency * theta_dot[branch] * q[mask] + frequency * q_dot[mask]
    theta_hvp_fd = (theta_plus - theta_minus) / (2.0 * h)
    x_hvp_fd = (x_bar_plus - x_bar_minus) / (2.0 * h)
    hvp_error = max(float(np.max(np.abs(theta_hvp - theta_hvp_fd))),
                    float(np.max(np.abs(x_hvp - x_hvp_fd))))
    value_error = float(np.max(np.abs(value - evaluate(x, theta))))
    derivative_error = float(np.max(np.abs(tangent - finite_difference)))
    return value_error, derivative_error, float(adjoint_error), hvp_error, float(np.max(np.abs(value)))


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def parse(stdout: str) -> dict[str, float | int | str]:
    values: dict[str, float | int | str] = {}
    for line in stdout.splitlines():
        if line.startswith("conditional_pipeline_transform_seconds,"):
            values["seconds"] = float(line.split(",", 1)[1])
        elif line.startswith("conditional_pipeline_route_error,"):
            values["route_error"] = float(line.split(",", 1)[1])
        elif line.startswith("conditional_pipeline_branch_count,"):
            values["branches"] = int(line.split(",", 1)[1])
        elif line.startswith("conditional_pipeline_feature_count,"):
            values["features"] = int(line.split(",", 1)[1])
        elif line.startswith("conditional_pipeline_cuda,"):
            values["cuda"] = line.split(",", 1)[1].strip()
    required = {"seconds", "route_error", "branches", "features", "cuda"}
    if set(values) != required:
        raise RuntimeError(f"release app omitted conditional metrics: {sorted(values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/conditional_pipeline.csv"))
    parser.add_argument("--target", default="fortml_bench_conditional_pipeline")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    value_error, derivative_error, adjoint_error, hvp_error, _ = independent_oracle()
    oracle_error = max(value_error, derivative_error, adjoint_error, hvp_error)
    if derivative_error > 2.0e-9 or adjoint_error > 2.0e-11 or hvp_error > 2.0e-5:
        raise RuntimeError(
            "independent conditional oracle failed: "
            f"jvp={derivative_error:.3e}, adjoint={adjoint_error:.3e}, "
            f"hvp={hvp_error:.3e}"
        )
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                   env=environment, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    focused = subprocess.run(
        ["fo", "exec", "--no-build", "test_conditional_pipeline"], cwd=fortml,
        env=environment, capture_output=True, text=True,
    )
    if focused.returncode != 0:
        raise RuntimeError("conditional focused oracle failed:\n" +
                           (focused.stdout + focused.stderr)[-4000:])
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    observed = parse(completed.stdout)
    if (observed["branches"] != 2 or observed["features"] != N_FEATURES or
            float(observed["route_error"]) > 2.0e-13 or observed["cuda"] != "unavailable"):
        raise RuntimeError(f"conditional release contract mismatch: {observed}")
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml, (
            fortml / "test_mlp_amsgrad_checkpoint.txt",
            fortml / "test_mlp_radam_checkpoint.txt",
        )),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    rows = [
        row(details, workload="conditional_basis_pipeline", phase="value_derivatives",
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_branches=2,
            seconds_per_operation=observed["seconds"], metric="oracle_max_abs_error",
            value=1.0, max_abs_error=oracle_error,
            oracle="independent NumPy interval-routed Fourier value/JVP/VJP/HVP oracle",
            notes="focused Fortran finite-difference/adjoint test plus release workload"),
        row(details, workload="conditional_basis_pipeline", phase="device_contract",
            backend="fortml", device="cuda", status="unavailable", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_branches=2, metric="api_surface", value="unavailable",
            max_abs_error=0.0, oracle="typed CUDA refusal preserves outputs",
            notes="resident route-mask CUDA executor is not linked"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
