#!/usr/bin/env python3
"""Benchmark exact-GP and MLP products against independent NumPy oracles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


GP_N = 128
GP_D = 4
GP_OUTPUTS = 2
GP_TEST_N = 32
GP_FIT_REPETITIONS = 8
GP_PREDICT_REPETITIONS = 32
GP_SIGNAL_VARIANCE = 1.4
GP_LENGTHSCALE = 0.9
GP_NOISE_VARIANCE = 0.08
GP_JITTER = 1.0e-10

MLP_N = 512
MLP_D = 16
MLP_HIDDEN = 32
MLP_OUTPUTS = 4
MLP_FORWARD_REPETITIONS = 64
MLP_VJP_REPETITIONS = 32

FIELDS = (
    "workload",
    "phase",
    "backend",
    "device",
    "status",
    "n_samples",
    "n_features",
    "n_hidden",
    "n_outputs",
    "n_test",
    "dtype",
    "threads",
    "cpu_affinity",
    "repetitions",
    "warmups",
    "seconds_per_operation",
    "setup_seconds",
    "build_seconds",
    "peak_rss_kib",
    "peak_device_bytes",
    "executable_bytes",
    "max_abs_error",
    "correctness_oracle",
    "compiler",
    "compiler_version",
    "flags",
    "python_version",
    "numpy_version",
    "torch_version",
    "gpytorch_version",
    "cuda_version",
    "cpu_model",
    "gpu_model",
    "driver_version",
    "os",
    "fortml_revision",
    "fortnum_revision",
    "benchmark_revision",
    "fortml_diff_sha256",
    "driver_sha256",
    "app_sha256",
    "notes",
)


def git_revision(repository: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip()
    return revision + ("+dirty" if dirty else "")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_diff_sha256(repository: Path) -> str:
    difference = subprocess.check_output(
        ["git", "-C", str(repository), "diff", "--binary", "HEAD"]
    )
    return hashlib.sha256(difference).hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def gpu_metadata() -> tuple[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        line = subprocess.check_output(command, text=True).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return "unavailable", "unavailable"
    name, driver = (item.strip() for item in line.split(",", 1))
    return name, driver


def common_metadata(root: Path, fortml: Path, cpu: int) -> dict[str, Any]:
    gpu, driver = gpu_metadata()
    return {
        "dtype": "float64",
        "threads": 1,
        "cpu_affinity": cpu,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_model": cpu_model(),
        "gpu_model": gpu,
        "driver_version": driver,
        "os": platform.platform(),
        "fortml_revision": git_revision(fortml),
        "fortnum_revision": git_revision(fortml.parent / "fortnum"),
        "benchmark_revision": git_revision(root),
        "fortml_diff_sha256": tracked_diff_sha256(fortml),
        "driver_sha256": sha256(Path(__file__).resolve()),
    }


def gp_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(1, GP_N + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, GP_D + 1, dtype=np.float64)[None, :]
    x = np.sin(0.013 * (rows + 3.0 * columns))
    x += 0.1 * np.cos(0.017 * rows * columns)

    test_rows = np.arange(1, GP_TEST_N + 1, dtype=np.float64)[:, None]
    x_test = np.sin(0.019 * (test_rows + 2.0 * columns))
    x_test += 0.1 * np.cos(0.011 * test_rows * columns)

    outputs = np.arange(1, GP_OUTPUTS + 1, dtype=np.float64)[None, :]
    y = np.sin(0.021 * rows * outputs)
    y += 0.3 * np.cos(0.007 * (rows + 2.0 * outputs))
    return x, y, x_test


def rbf(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = left[:, None, :] - right[None, :, :]
    squared_distance = np.sum(difference * difference, axis=2)
    return GP_SIGNAL_VARIANCE * np.exp(
        -0.5 * squared_distance / (GP_LENGTHSCALE * GP_LENGTHSCALE)
    )


def gp_oracle() -> dict[str, np.ndarray]:
    x, y, x_test = gp_inputs()
    covariance = rbf(x, x)
    covariance.flat[:: GP_N + 1] += GP_NOISE_VARIANCE + GP_JITTER
    cross = rbf(x, x_test)
    alpha = np.linalg.solve(covariance, y)
    solved_cross = np.linalg.solve(covariance, cross)
    mean = cross.T @ alpha
    variance = GP_SIGNAL_VARIANCE - np.sum(cross * solved_cross, axis=0)
    return {
        "x": x,
        "y": y,
        "x_test": x_test,
        "alpha": alpha,
        "mean": mean,
        "variance": variance,
    }


def mlp_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(1, MLP_N + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, MLP_D + 1, dtype=np.float64)[None, :]
    x = np.sin(0.013 * rows + 0.071 * columns)
    x += np.cos(0.009 * rows * columns)

    n_parameters = MLP_D * MLP_HIDDEN + MLP_HIDDEN
    n_parameters += MLP_HIDDEN * MLP_OUTPUTS + MLP_OUTPUTS
    indices = np.arange(1, n_parameters + 1, dtype=np.float64)
    theta = 0.08 * np.sin(0.37 * indices)

    outputs = np.arange(1, MLP_OUTPUTS + 1, dtype=np.float64)[None, :]
    u = 0.2 * np.sin(0.011 * rows * outputs)
    return x, theta, u


def unpack_mlp(theta: np.ndarray) -> tuple[np.ndarray, ...]:
    position = 0
    count = MLP_D * MLP_HIDDEN
    weight_1 = theta[position : position + count].reshape(
        (MLP_D, MLP_HIDDEN), order="F"
    )
    position += count
    bias_1 = theta[position : position + MLP_HIDDEN]
    position += MLP_HIDDEN
    count = MLP_HIDDEN * MLP_OUTPUTS
    weight_2 = theta[position : position + count].reshape(
        (MLP_HIDDEN, MLP_OUTPUTS), order="F"
    )
    position += count
    bias_2 = theta[position : position + MLP_OUTPUTS]
    return weight_1, bias_1, weight_2, bias_2


def mlp_oracle() -> dict[str, np.ndarray]:
    x, theta, u = mlp_inputs()
    weight_1, bias_1, weight_2, bias_2 = unpack_mlp(theta)
    hidden = np.tanh(x @ weight_1 + bias_1)
    prediction = hidden @ weight_2 + bias_2
    hidden_bar = u @ weight_2.T
    preactivation_bar = hidden_bar * (1.0 - hidden * hidden)
    weight_1_bar = x.T @ preactivation_bar
    bias_1_bar = np.sum(preactivation_bar, axis=0)
    weight_2_bar = hidden.T @ u
    bias_2_bar = np.sum(u, axis=0)
    parameter_bar = np.concatenate(
        (
            weight_1_bar.reshape(-1, order="F"),
            bias_1_bar,
            weight_2_bar.reshape(-1, order="F"),
            bias_2_bar,
        )
    )
    input_bar = preactivation_bar @ weight_1.T
    return {
        "x": x,
        "theta": theta,
        "u": u,
        "prediction": prediction,
        "parameter_bar": parameter_bar,
        "input_bar": input_bar,
    }


def read_fortran_oracle(path: Path, workload: str) -> dict[str, np.ndarray]:
    if workload == "exact_gp":
        arrays = {
            "mean": np.full((GP_TEST_N, GP_OUTPUTS), np.nan),
            "variance": np.full((GP_TEST_N,), np.nan),
        }
    else:
        n_parameters = MLP_D * MLP_HIDDEN + MLP_HIDDEN
        n_parameters += MLP_HIDDEN * MLP_OUTPUTS + MLP_OUTPUTS
        arrays = {
            "prediction": np.full((MLP_N, MLP_OUTPUTS), np.nan),
            "parameter_bar": np.full((n_parameters,), np.nan),
            "input_bar": np.full((MLP_N, MLP_D), np.nan),
        }
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            quantity = row["quantity"]
            first = int(row["row"]) - 1
            second = int(row["column"]) - 1
            value = float(row["value"])
            target = arrays[quantity]
            if target.ndim == 1:
                target[first] = value
            else:
                target[first, second] = value
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError(f"incomplete Fortran {workload} oracle output")
    return arrays


def fortran_error(workload: str, actual: dict[str, np.ndarray]) -> tuple[float, float]:
    if workload == "exact_gp":
        expected = gp_oracle()
        forward_error = float(np.max(np.abs(actual["mean"] - expected["mean"])))
        variance_error = float(
            np.max(np.abs(actual["variance"] - expected["variance"]))
        )
        return max(forward_error, variance_error), max(forward_error, variance_error)
    expected = mlp_oracle()
    forward_error = float(
        np.max(np.abs(actual["prediction"] - expected["prediction"]))
    )
    parameter_error = np.max(
        np.abs(actual["parameter_bar"] - expected["parameter_bar"])
    )
    input_error = np.max(np.abs(actual["input_bar"] - expected["input_bar"]))
    return forward_error, float(max(parameter_error, input_error))


def executable_size(fortml: Path, target: str) -> int:
    candidates = [
        fortml / "build" / "fo" / "bin" / target,
        fortml / "build" / "fo" / "app" / target,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.stat().st_size
    raise RuntimeError(f"fo did not produce {target}")


def compiler_version(compiler: str) -> str:
    output = subprocess.check_output(
        [compiler, "--version"], text=True, stderr=subprocess.STDOUT
    )
    return next(line.strip() for line in output.splitlines() if line.strip())


def time_fortran_target(
    fortml: Path, target: str, environment: dict[str, str]
) -> tuple[str, int]:
    marker = "__FORTML_BENCH_MAX_RSS_KIB__="
    command = [
        "/usr/bin/time",
        "-f",
        marker + "%M",
        "fo",
        "exec",
        "--no-build",
        target,
    ]
    completed = subprocess.run(
        command,
        cwd=fortml,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(re.escape(marker) + r"(\d+)", completed.stderr)
    if match is None:
        raise RuntimeError(f"missing peak RSS from {target}")
    return completed.stdout, int(match.group(1))


def parse_phase_rows(stdout: str, expected: set[str]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] not in expected:
            continue
        if len(fields) != 6:
            raise RuntimeError(f"unexpected benchmark row: {line}")
        parsed[fields[0]] = {
            "n_samples": int(fields[1]),
            "n_features": int(fields[2]),
            "n_outputs": int(fields[3]),
            "repetitions": int(fields[4]),
            "seconds_per_operation": float(fields[5]),
        }
    missing = expected - set(parsed)
    if missing:
        raise RuntimeError(f"missing benchmark phases: {sorted(missing)}")
    return parsed


def unavailable_fortran_rows(
    metadata: dict[str, Any], backend: str, status: str, note: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phases = (
        ("exact_gp", "fit", GP_N, GP_D, 0, GP_OUTPUTS, GP_TEST_N, GP_FIT_REPETITIONS),
        (
            "exact_gp",
            "predict",
            GP_N,
            GP_D,
            0,
            GP_OUTPUTS,
            GP_TEST_N,
            GP_PREDICT_REPETITIONS,
        ),
        (
            "mlp",
            "forward",
            MLP_N,
            MLP_D,
            MLP_HIDDEN,
            MLP_OUTPUTS,
            0,
            MLP_FORWARD_REPETITIONS,
        ),
        (
            "mlp",
            "vjp",
            MLP_N,
            MLP_D,
            MLP_HIDDEN,
            MLP_OUTPUTS,
            0,
            MLP_VJP_REPETITIONS,
        ),
    )
    for workload, phase, n, d, hidden, outputs, n_test, repetitions in phases:
        row = dict(metadata)
        row.update(
            {
                "workload": workload,
                "phase": phase,
                "backend": backend,
                "device": "cpu",
                "status": status,
                "n_samples": n,
                "n_features": d,
                "n_hidden": hidden,
                "n_outputs": outputs,
                "n_test": n_test,
                "repetitions": repetitions,
                "warmups": 1,
                "notes": note,
            }
        )
        rows.append(row)
    return rows


def benchmark_fortran(
    root: Path,
    fortml: Path,
    compiler: str,
    flags: str,
    cpu: int,
    include_cuda_boundary: bool,
) -> list[dict[str, Any]]:
    metadata = common_metadata(root, fortml, cpu)
    backend = f"fortml_{Path(compiler).name}"
    metadata.update(
        {
            "backend": backend,
            "compiler": compiler,
            "flags": flags,
            "correctness_oracle": "independent_numpy_full_output",
        }
    )
    if not Path(compiler).is_file() and not shutil_which(compiler):
        return unavailable_fortran_rows(metadata, backend, "unavailable", "compiler not found")

    environment = os.environ.copy()
    environment.update(
        {
            "FO_FC": compiler,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "build", "--flag", flags],
        cwd=fortml,
        env=environment,
        capture_output=True,
        text=True,
    )
    build_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        note = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "fo build failed"
        return unavailable_fortran_rows(metadata, backend, "build_failed", note)

    version_text = compiler_version(compiler)
    rows: list[dict[str, Any]] = []
    workloads = (
        (
            "exact_gp",
            "fortml_bench_gp",
            {"gp_fit", "gp_predict"},
            fortml / "app" / "fortml_bench_gp.f90",
        ),
        (
            "mlp",
            "fortml_bench_mlp",
            {"mlp_forward", "mlp_vjp"},
            fortml / "app" / "fortml_bench_mlp.f90",
        ),
    )
    for workload, target, phase_names, app_path in workloads:
        with tempfile.TemporaryDirectory() as directory:
            oracle_path = Path(directory) / "oracle.csv"
            oracle_environment = environment.copy()
            oracle_environment.update(
                {
                    "FORTML_BENCH_ORACLE": str(oracle_path),
                    "FORTML_BENCH_ORACLE_ONLY": "1",
                }
            )
            subprocess.run(
                ["fo", "exec", "--no-build", target],
                cwd=fortml,
                env=oracle_environment,
                capture_output=True,
                text=True,
                check=True,
            )
            actual = read_fortran_oracle(oracle_path, workload)
            first_error, second_error = fortran_error(workload, actual)
            tolerance = 5.0e-10 if workload == "exact_gp" else 5.0e-11
            if max(first_error, second_error) > tolerance:
                raise RuntimeError(
                    f"{backend} {workload} NumPy oracle mismatch: "
                    f"{max(first_error, second_error):.3e}"
                )

        stdout, peak_rss = time_fortran_target(fortml, target, environment)
        phases = parse_phase_rows(stdout, phase_names)
        phase_map = (
            (("gp_fit", "fit", first_error), ("gp_predict", "predict", second_error))
            if workload == "exact_gp"
            else (
                ("mlp_forward", "forward", first_error),
                ("mlp_vjp", "vjp", second_error),
            )
        )
        for source_phase, phase, error in phase_map:
            measured = phases[source_phase]
            row = dict(metadata)
            row.update(measured)
            row.update(
                {
                    "workload": workload,
                    "phase": phase,
                    "device": "cpu",
                    "status": "pass",
                    "n_hidden": MLP_HIDDEN if workload == "mlp" else 0,
                    "n_test": GP_TEST_N if workload == "exact_gp" else 0,
                    "warmups": 1,
                    "build_seconds": build_seconds,
                    "peak_rss_kib": peak_rss,
                    "peak_device_bytes": 0,
                    "executable_bytes": executable_size(fortml, target),
                    "max_abs_error": error,
                    "compiler_version": version_text,
                    "app_sha256": sha256(app_path),
                    "notes": "serial host implementation; release fo build",
                }
            )
            rows.append(row)

        if include_cuda_boundary:
            for _, phase, _ in phase_map:
                row = dict(metadata)
                row.update(
                    {
                        "workload": workload,
                        "phase": phase,
                        "device": "cuda",
                        "status": "unsupported",
                        "n_samples": GP_N if workload == "exact_gp" else MLP_N,
                        "n_features": GP_D if workload == "exact_gp" else MLP_D,
                        "n_hidden": MLP_HIDDEN if workload == "mlp" else 0,
                        "n_outputs": GP_OUTPUTS if workload == "exact_gp" else MLP_OUTPUTS,
                        "n_test": GP_TEST_N if workload == "exact_gp" else 0,
                        "repetitions": 0,
                        "warmups": 0,
                        "build_seconds": build_seconds,
                        "executable_bytes": executable_size(fortml, target),
                        "compiler_version": version_text,
                        "app_sha256": sha256(app_path),
                        "notes": "FortML exact-GP and MLP apps have no device-resident implementation",
                    }
                )
                rows.append(row)
    return rows


def shutil_which(command: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def torch_peak(torch: Any, device: Any) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def worker_gp(root: Path, fortml: Path, device_name: str, cpu: int) -> list[dict[str, Any]]:
    import torch
    import gpytorch

    metadata = common_metadata(root, fortml, cpu)
    metadata.update(
        {
            "backend": "gpytorch_exact",
            "compiler": "python",
            "flags": "torch.set_num_threads(1)",
            "torch_version": torch.__version__,
            "gpytorch_version": version("gpytorch"),
            "cuda_version": torch.version.cuda or "unavailable",
            "correctness_oracle": "independent_numpy_dense_solve",
            "app_sha256": "not_applicable",
        }
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        rows = []
        for phase in ("fit", "predict"):
            row = dict(metadata)
            row.update(
                {
                    "workload": "exact_gp",
                    "phase": phase,
                    "device": "cuda",
                    "status": "unavailable",
                    "n_samples": GP_N,
                    "n_features": GP_D,
                    "n_hidden": 0,
                    "n_outputs": GP_OUTPUTS,
                    "n_test": GP_TEST_N,
                    "notes": "torch.cuda.is_available() is false",
                }
            )
            rows.append(row)
        return rows

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(device_name)
    oracle = gp_oracle()
    started = time.perf_counter()
    x = torch.as_tensor(oracle["x"], dtype=torch.float64, device=device)
    y = torch.as_tensor(oracle["y"], dtype=torch.float64, device=device)
    x_test = torch.as_tensor(oracle["x_test"], dtype=torch.float64, device=device)
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=GP_D)
    ).to(device=device, dtype=torch.float64)
    kernel.base_kernel.lengthscale = GP_LENGTHSCALE
    kernel.outputscale = GP_SIGNAL_VARIANCE
    setup_seconds = time.perf_counter() - started

    def fit() -> tuple[Any, Any]:
        covariance = kernel(x, x).to_dense()
        covariance = covariance + (GP_NOISE_VARIANCE + GP_JITTER) * torch.eye(
            GP_N, dtype=torch.float64, device=device
        )
        factor = torch.linalg.cholesky(covariance)
        alpha = torch.cholesky_solve(y, factor)
        return factor, alpha

    def predict(factor: Any, alpha: Any) -> tuple[Any, Any]:
        cross = kernel(x, x_test).to_dense()
        prior = kernel(x_test, x_test).to_dense()
        solved_cross = torch.cholesky_solve(cross, factor)
        mean = cross.transpose(0, 1) @ alpha
        variance = torch.diagonal(prior) - torch.sum(cross * solved_cross, dim=0)
        return mean, variance

    with torch.no_grad():
        factor, alpha = fit()
        mean, variance = predict(factor, alpha)
        synchronize(torch, device)
        predict_error = max(
            float(np.max(np.abs(mean.cpu().numpy() - oracle["mean"]))),
            float(np.max(np.abs(variance.cpu().numpy() - oracle["variance"]))),
        )
        # The training covariance is deliberately ill-conditioned. Validate
        # the fit through its posterior rather than comparing unstable alpha
        # coordinates from LU and Cholesky solves.
        fit_error = predict_error
        if not np.isfinite([fit_error, predict_error]).all() or max(
            fit_error, predict_error
        ) > 1.0e-7:
            raise RuntimeError(
                f"GPyTorch {device_name} NumPy oracle mismatch: "
                f"{max(fit_error, predict_error):.3e}"
            )

        for _ in range(2):
            factor, alpha = fit()
        synchronize(torch, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(GP_FIT_REPETITIONS):
            factor, alpha = fit()
        synchronize(torch, device)
        fit_seconds = (time.perf_counter() - started) / GP_FIT_REPETITIONS
        fit_peak_device = torch_peak(torch, device)

        for _ in range(2):
            predict(factor, alpha)
        synchronize(torch, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(GP_PREDICT_REPETITIONS):
            predict(factor, alpha)
        synchronize(torch, device)
        predict_seconds = (time.perf_counter() - started) / GP_PREDICT_REPETITIONS
        predict_peak_device = torch_peak(torch, device)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rows = []
    for phase, repetitions, seconds, error, peak_device in (
        ("fit", GP_FIT_REPETITIONS, fit_seconds, fit_error, fit_peak_device),
        (
            "predict",
            GP_PREDICT_REPETITIONS,
            predict_seconds,
            predict_error,
            predict_peak_device,
        ),
    ):
        row = dict(metadata)
        row.update(
            {
                "workload": "exact_gp",
                "phase": phase,
                "device": device_name,
                "status": "pass",
                "n_samples": GP_N,
                "n_features": GP_D,
                "n_hidden": 0,
                "n_outputs": GP_OUTPUTS,
                "n_test": GP_TEST_N,
                "repetitions": repetitions,
                "warmups": 2,
                "seconds_per_operation": seconds,
                "setup_seconds": setup_seconds,
                "build_seconds": 0.0,
                "peak_rss_kib": peak_rss,
                "peak_device_bytes": peak_device,
                "executable_bytes": "not_applicable",
                "max_abs_error": error,
                "notes": "GPyTorch RBF kernel with matched dense Cholesky and solves",
            }
        )
        rows.append(row)
    return rows


def worker_mlp(root: Path, fortml: Path, device_name: str, cpu: int) -> list[dict[str, Any]]:
    import torch

    metadata = common_metadata(root, fortml, cpu)
    metadata.update(
        {
            "backend": "pytorch",
            "compiler": "python",
            "flags": "torch.set_num_threads(1)",
            "torch_version": torch.__version__,
            "gpytorch_version": version("gpytorch"),
            "cuda_version": torch.version.cuda or "unavailable",
            "correctness_oracle": "independent_numpy_explicit_forward_and_vjp",
            "app_sha256": "not_applicable",
        }
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        rows = []
        for phase in ("forward", "vjp"):
            row = dict(metadata)
            row.update(
                {
                    "workload": "mlp",
                    "phase": phase,
                    "device": "cuda",
                    "status": "unavailable",
                    "n_samples": MLP_N,
                    "n_features": MLP_D,
                    "n_hidden": MLP_HIDDEN,
                    "n_outputs": MLP_OUTPUTS,
                    "n_test": 0,
                    "notes": "torch.cuda.is_available() is false",
                }
            )
            rows.append(row)
        return rows

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(device_name)
    oracle = mlp_oracle()
    started = time.perf_counter()
    x = torch.as_tensor(oracle["x"], dtype=torch.float64, device=device)
    theta = torch.as_tensor(oracle["theta"], dtype=torch.float64, device=device)
    u = torch.as_tensor(oracle["u"], dtype=torch.float64, device=device)
    setup_seconds = time.perf_counter() - started

    def forward(parameters: Any, inputs: Any) -> Any:
        position = 0
        count = MLP_D * MLP_HIDDEN
        weight_1 = parameters[position : position + count].reshape(
            MLP_HIDDEN, MLP_D
        ).transpose(0, 1)
        position += count
        bias_1 = parameters[position : position + MLP_HIDDEN]
        position += MLP_HIDDEN
        count = MLP_HIDDEN * MLP_OUTPUTS
        weight_2 = parameters[position : position + count].reshape(
            MLP_OUTPUTS, MLP_HIDDEN
        ).transpose(0, 1)
        position += count
        bias_2 = parameters[position : position + MLP_OUTPUTS]
        hidden = torch.tanh(inputs @ weight_1 + bias_1)
        return hidden @ weight_2 + bias_2

    def vjp() -> tuple[Any, Any]:
        active_theta = theta.detach().requires_grad_(True)
        active_x = x.detach().requires_grad_(True)
        prediction = forward(active_theta, active_x)
        return torch.autograd.grad(prediction, (active_theta, active_x), u)

    with torch.no_grad():
        prediction = forward(theta, x)
    parameter_bar, input_bar = vjp()
    synchronize(torch, device)
    forward_error = float(
        np.max(np.abs(prediction.detach().cpu().numpy() - oracle["prediction"]))
    )
    vjp_error = max(
        float(
            np.max(
                np.abs(
                    parameter_bar.detach().cpu().numpy() - oracle["parameter_bar"]
                )
            )
        ),
        float(
            np.max(
                np.abs(input_bar.detach().cpu().numpy() - oracle["input_bar"])
            )
        ),
    )
    if not np.isfinite([forward_error, vjp_error]).all() or max(
        forward_error, vjp_error
    ) > 5.0e-11:
        raise RuntimeError(
            f"PyTorch {device_name} NumPy oracle mismatch: "
            f"{max(forward_error, vjp_error):.3e}"
        )

    with torch.no_grad():
        for _ in range(2):
            forward(theta, x)
    synchronize(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(MLP_FORWARD_REPETITIONS):
            forward(theta, x)
    synchronize(torch, device)
    forward_seconds = (time.perf_counter() - started) / MLP_FORWARD_REPETITIONS
    forward_peak_device = torch_peak(torch, device)

    for _ in range(2):
        vjp()
    synchronize(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(MLP_VJP_REPETITIONS):
        vjp()
    synchronize(torch, device)
    vjp_seconds = (time.perf_counter() - started) / MLP_VJP_REPETITIONS
    vjp_peak_device = torch_peak(torch, device)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rows = []
    for phase, repetitions, seconds, error, peak_device in (
        (
            "forward",
            MLP_FORWARD_REPETITIONS,
            forward_seconds,
            forward_error,
            forward_peak_device,
        ),
        ("vjp", MLP_VJP_REPETITIONS, vjp_seconds, vjp_error, vjp_peak_device),
    ):
        row = dict(metadata)
        row.update(
            {
                "workload": "mlp",
                "phase": phase,
                "device": device_name,
                "status": "pass",
                "n_samples": MLP_N,
                "n_features": MLP_D,
                "n_hidden": MLP_HIDDEN,
                "n_outputs": MLP_OUTPUTS,
                "n_test": 0,
                "repetitions": repetitions,
                "warmups": 2,
                "seconds_per_operation": seconds,
                "setup_seconds": setup_seconds,
                "build_seconds": 0.0,
                "peak_rss_kib": peak_rss,
                "peak_device_bytes": peak_device,
                "executable_bytes": "not_applicable",
                "max_abs_error": error,
                "notes": "explicit tanh network; VJP includes forward graph construction",
            }
        )
        rows.append(row)
    return rows


def run_worker(
    root: Path, fortml: Path, workload: str, device: str, cpu: int
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "rows.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            workload,
            "--device",
            device,
            "--worker-output",
            str(output),
            "--fortml",
            str(fortml),
            "--cpu",
            str(cpu),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        subprocess.run(command, cwd=root, env=environment, check=True)
        return json.loads(output.read_text())


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("status") == "pass":
        for field in (
            "seconds_per_operation",
            "max_abs_error",
            "repetitions",
            "warmups",
            "peak_rss_kib",
        ):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"pass row contains invalid {field}") from error
            if not np.isfinite(value) or value < 0.0:
                raise RuntimeError(f"pass row contains nonfinite or negative {field}")
        if float(row["seconds_per_operation"]) <= 0.0:
            raise RuntimeError("pass row contains nonpositive timing")
        if float(row["repetitions"]) < 1.0 or float(row["peak_rss_kib"]) <= 0.0:
            raise RuntimeError("pass row contains invalid repetitions or peak RSS")
        if row.get("device") == "cuda":
            try:
                peak_device = float(row["peak_device_bytes"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("CUDA pass row lacks a device peak") from error
            if not np.isfinite(peak_device) or peak_device <= 0.0:
                raise RuntimeError("CUDA pass row contains an invalid device peak")
    return {field: row.get(field, "") for field in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/model_workloads.csv")
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("cpu", "cuda"),
        default=("cpu", "cuda"),
    )
    parser.add_argument(
        "--compilers", nargs="+", default=("gfortran", "nvfortran")
    )
    parser.add_argument("--gfortran-flags", default="-O3 -march=native")
    parser.add_argument("--nvfortran-flags", default="-O3 -mp=multicore")
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--worker", choices=("exact_gp", "mlp"), default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--worker-output", type=Path, default=None)
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    fortml = arguments.fortml.resolve()
    allowed_cpus = sorted(os.sched_getaffinity(0))
    cpu = arguments.cpu if arguments.cpu is not None else allowed_cpus[0]
    if cpu not in allowed_cpus:
        raise SystemExit(f"CPU {cpu} is outside this process's affinity mask")
    os.sched_setaffinity(0, {cpu})

    if arguments.worker is not None:
        if arguments.worker_output is None:
            raise SystemExit("--worker-output is required with --worker")
        if arguments.worker == "exact_gp":
            rows = worker_gp(root, fortml, arguments.device, cpu)
        else:
            rows = worker_mlp(root, fortml, arguments.device, cpu)
        arguments.worker_output.write_text(json.dumps(rows))
        return

    rows: list[dict[str, Any]] = []
    for compiler in arguments.compilers:
        name = Path(compiler).name
        flags = (
            arguments.nvfortran_flags
            if name == "nvfortran"
            else arguments.gfortran_flags
        )
        rows.extend(
            benchmark_fortran(
                root,
                fortml,
                compiler,
                flags,
                cpu,
                "cuda" in arguments.devices,
            )
        )
    for device in arguments.devices:
        rows.extend(run_worker(root, fortml, "exact_gp", device, cpu))
        rows.extend(run_worker(root, fortml, "mlp", device, cpu))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(normalize_row(row) for row in rows)
    print(arguments.output)


if __name__ == "__main__":
    main()
