"""Benchmark FortML's spectral, derivative, multi-output, and variational GPs.

Every Python and FortML path is checked against a full output assembled by an
independent NumPy implementation before timing. The FortML executable also
retains its internal direct-formula checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch


SIGNAL = 1.4
SCALE = 0.9
SHIFT = 0.35
WORKLOADS = (
    "logdet",
    "predictive_variance",
    "derivative",
    "multi_output",
    "variational",
)
FORTRAN_FIELDS = (
    "workload",
    "n_samples",
    "n_features",
    "n_outputs",
    "n_test",
    "n_inducing",
    "repetitions",
    "seconds_per_call",
    "relative_error",
)
CSV_FIELDS = FORTRAN_FIELDS + (
    "backend",
    "device",
    "status",
    "warmups",
    "setup_seconds",
    "build_seconds",
    "peak_memory_bytes",
    "executable_bytes",
    "oracle",
    "threads",
    "cpu_affinity",
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
    "lanczos_source_sha256",
    "notes",
)
WORKER_FIELDS = (
    "workload",
    "device",
    "backend",
    "status",
    "repetitions",
    "warmups",
    "seconds_per_call",
    "relative_error",
    "setup_seconds",
    "peak_memory_bytes",
)


def deterministic_points(n: int, d: int) -> np.ndarray:
    rows = np.arange(1, n + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, d + 1, dtype=np.float64)[None, :]
    return 0.4 * np.sin(rows + 3.0 * columns) + 0.2 * np.cos(2.0 * rows - columns)


def deterministic_queries(n: int, d: int) -> np.ndarray:
    rows = np.arange(1, n + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, d + 1, dtype=np.float64)[None, :]
    return 0.31 * np.sin(2.0 * rows + columns) - 0.17 * np.cos(rows - 2.0 * columns)


def rbf(left: np.ndarray, right: np.ndarray, signal: float, scale: float) -> np.ndarray:
    difference = left[:, None, :] - right[None, :, :]
    return signal * np.exp(-0.5 * np.sum(difference * difference, axis=2) / scale**2)


def torch_rbf(
    left: torch.Tensor, right: torch.Tensor, signal: float, scale: float
) -> torch.Tensor:
    difference = left[:, None, :] - right[None, :, :]
    return signal * torch.exp(-0.5 * (difference * difference).sum(dim=2) / scale**2)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def versions() -> dict[str, str]:
    def package(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gpytorch_version": package("gpytorch"),
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda or "unavailable",
    }


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


def compiler_version(compiler: str) -> str:
    completed = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True
    )
    return completed.stdout.splitlines()[0].strip()


def build_fortml(fortml: Path, compiler: str, flags: str) -> float:
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
    subprocess.run(
        ["fo", "build", "--flag", flags],
        cwd=fortml,
        env=environment,
        check=True,
        capture_output=True,
    )
    return time.perf_counter() - started


def executable_size(fortml: Path) -> int:
    target = "fortml_bench_gp_features"
    for candidate in (
        fortml / "build" / "fo" / "bin" / target,
        fortml / "build" / "fo" / "app" / target,
    ):
        if candidate.is_file():
            return candidate.stat().st_size
    raise RuntimeError(f"fo did not produce {target}")


def run_fortml(
    fortml: Path,
    workload: str,
    repetitions: int,
    compiler: str,
    expected: np.ndarray,
) -> tuple[dict[str, str], int, float]:
    environment = os.environ.copy()
    environment.update(
        {
            "FO_FC": compiler,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
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
            [
                "fo",
                "exec",
                "--no-build",
                "fortml_bench_gp_features",
                workload,
                str(repetitions),
            ],
            cwd=fortml,
            env=oracle_environment,
            text=True,
            capture_output=True,
            check=True,
        )
        with oracle_path.open(newline="") as stream:
            actual = np.array(
                [float(row["value"]) for row in csv.DictReader(stream)]
            )
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"FortML {workload} oracle shape {actual.shape} != {expected.shape}"
        )
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError(f"FortML {workload} oracle contains a nonfinite value")
    external_error = float(
        np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1.0)
    )
    tolerance = 0.05 if workload in {"logdet", "predictive_variance"} else 3.0e-9
    if not np.isfinite(external_error) or external_error > tolerance:
        raise RuntimeError(
            f"FortML {workload} external NumPy oracle mismatch: {external_error:g}"
        )

    marker = "__FORTML_GP_FEATURES_MAX_RSS_KIB__="
    completed = subprocess.run(
        [
            "/usr/bin/time",
            "-f",
            marker + "%M",
            "fo",
            "exec",
            "--no-build",
            "fortml_bench_gp_features",
            workload,
            str(repetitions),
        ],
        cwd=fortml,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    peak_match = next(
        (
            line.removeprefix(marker)
            for line in completed.stderr.splitlines()
            if line.startswith(marker)
        ),
        None,
    )
    if peak_match is None:
        raise RuntimeError("FortML peak RSS marker is missing")
    line = completed.stdout.strip().splitlines()[-1]
    values = [part.strip() for part in line.split(",")]
    if len(values) != len(FORTRAN_FIELDS):
        raise RuntimeError(f"unexpected FortML output: {line!r}")
    record = dict(zip(FORTRAN_FIELDS, values))
    n, d, outputs, n_test, n_inducing = shape(workload)
    expected_counts = (n, d, outputs, n_test, n_inducing, repetitions)
    try:
        actual_counts = tuple(
            int(record[field])
            for field in (
                "n_samples",
                "n_features",
                "n_outputs",
                "n_test",
                "n_inducing",
                "repetitions",
            )
        )
        seconds = float(record["seconds_per_call"])
        internal_error = float(record["relative_error"])
        peak = int(peak_match) * 1024
    except ValueError as error:
        raise RuntimeError(f"invalid FortML {workload} scalar output") from error
    if record["workload"] != workload or actual_counts != expected_counts:
        raise RuntimeError(f"FortML {workload} returned mismatched workload metadata")
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise RuntimeError(f"FortML {workload} returned an invalid timing")
    if not math.isfinite(internal_error) or internal_error < 0.0:
        raise RuntimeError(f"FortML {workload} returned an invalid internal error")
    if peak <= 0:
        raise RuntimeError(f"FortML {workload} returned an invalid peak RSS")
    return record, peak, external_error


def derivative_data() -> tuple[np.ndarray, ...]:
    n, n_test = 48, 24
    x = np.linspace(-2.4, 2.4, n)[:, None]
    components = (np.arange(1, n + 1) % 2).astype(np.int64)
    y = np.empty((n, 2), dtype=np.float64)
    values = components == 0
    y[values, 0] = np.sin(1.3 * x[values, 0])
    y[values, 1] = np.cos(0.7 * x[values, 0])
    y[~values, 0] = 1.3 * np.cos(1.3 * x[~values, 0])
    y[~values, 1] = -0.7 * np.sin(0.7 * x[~values, 0])
    query = np.linspace(-2.2, 2.2, n_test)[:, None]
    query_components = ((np.arange(1, n_test + 1) + 1) % 2).astype(np.int64)
    return x, components, y, query, query_components


def derivative_covariance(
    left: np.ndarray,
    left_component: np.ndarray,
    right: np.ndarray,
    right_component: np.ndarray,
    signal: float,
    scale: float,
) -> np.ndarray:
    difference = left[:, None, 0] - right[None, :, 0]
    inverse = 1.0 / scale**2
    base = signal * np.exp(-0.5 * difference**2 * inverse)
    output = base.copy()
    left_derivative = left_component[:, None] == 1
    right_derivative = right_component[None, :] == 1
    output[left_derivative & ~right_derivative] = (
        -difference * inverse * base
    )[left_derivative & ~right_derivative]
    output[~left_derivative & right_derivative] = (
        difference * inverse * base
    )[~left_derivative & right_derivative]
    both = left_derivative & right_derivative
    output[both] = ((inverse - difference**2 * inverse**2) * base)[both]
    return output


def torch_derivative_covariance(
    left: torch.Tensor,
    left_component: torch.Tensor,
    right: torch.Tensor,
    right_component: torch.Tensor,
    signal: float,
    scale: float,
) -> torch.Tensor:
    difference = left[:, None, 0] - right[None, :, 0]
    inverse = 1.0 / scale**2
    base = signal * torch.exp(-0.5 * difference**2 * inverse)
    left_derivative = left_component[:, None] == 1
    right_derivative = right_component[None, :] == 1
    return torch.where(
        left_derivative & right_derivative,
        (inverse - difference**2 * inverse**2) * base,
        torch.where(
            left_derivative,
            -difference * inverse * base,
            torch.where(right_derivative, difference * inverse * base, base),
        ),
    )


def numpy_derivative_reference() -> np.ndarray:
    x, components, y, query, query_components = derivative_data()
    matrix = derivative_covariance(x, components, x, components, 1.3, 0.7)
    matrix.flat[:: matrix.shape[0] + 1] += 0.08 + 1.0e-10
    cross = derivative_covariance(x, components, query, query_components, 1.3, 0.7)
    alpha = np.linalg.solve(matrix, y)
    solved = np.linalg.solve(matrix, cross)
    mean = cross.T @ alpha
    prior = np.diag(
        derivative_covariance(
            query, query_components, query, query_components, 1.3, 0.7
        )
    )
    variance = prior - np.sum(cross * solved, axis=0)
    return np.concatenate((mean.ravel(), variance))


def prepare_torch_derivative(
    device: torch.device,
) -> tuple[Callable[[], torch.Tensor], float]:
    started = time.perf_counter()
    arrays = derivative_data()
    x = torch.as_tensor(arrays[0], dtype=torch.float64, device=device)
    components = torch.as_tensor(arrays[1], device=device)
    y = torch.as_tensor(arrays[2], dtype=torch.float64, device=device)
    query = torch.as_tensor(arrays[3], dtype=torch.float64, device=device)
    query_components = torch.as_tensor(arrays[4], device=device)
    identity = torch.eye(48, dtype=x.dtype, device=device)
    synchronize(device)
    setup_seconds = time.perf_counter() - started

    def calculate() -> torch.Tensor:
        matrix = torch_derivative_covariance(
            x, components, x, components, 1.3, 0.7
        )
        matrix = matrix + (0.08 + 1.0e-10) * identity
        cross = torch_derivative_covariance(
            x, components, query, query_components, 1.3, 0.7
        )
        factor = torch.linalg.cholesky(matrix)
        alpha = torch.cholesky_solve(y, factor)
        solved = torch.cholesky_solve(cross, factor)
        mean = cross.T @ alpha
        prior = torch.diagonal(
            torch_derivative_covariance(
                query, query_components, query, query_components, 1.3, 0.7
            )
        )
        variance = prior - (cross * solved).sum(dim=0)
        return torch.cat((mean.reshape(-1), variance))

    return calculate, setup_seconds


def multi_output_data() -> tuple[np.ndarray, ...]:
    n, p, n_test = 24, 3, 12
    x = np.linspace(-1.8, 1.8, n)[:, None]
    y = np.column_stack(
        [np.sin((j + 1) * x[:, 0]) + 0.1 * (j + 1) * np.cos(0.4 * x[:, 0]) for j in range(p)]
    )
    query = np.linspace(-1.7, 1.7, n_test)[:, None]
    weights = np.array([[0.9, 0.2], [-0.4, 0.7], [0.6, -0.3]])
    independent = np.array([0.25, 0.35, 0.45])
    return x, y, query, weights, independent


def numpy_multi_output_reference() -> np.ndarray:
    x, y, query, weights, independent = multi_output_data()
    coreg = weights @ weights.T + np.diag(independent)
    joint = np.kron(coreg, rbf(x, x, 1.2, 0.75))
    joint.flat[:: joint.shape[0] + 1] += 0.12
    cross = np.kron(coreg, rbf(query, x, 1.2, 0.75))
    alpha = np.linalg.solve(joint, y.T.reshape(-1))
    return (cross @ alpha).reshape(3, 12).T.ravel()


def prepare_torch_multi_output(
    device: torch.device,
) -> tuple[Callable[[], torch.Tensor], float]:
    started = time.perf_counter()
    arrays = multi_output_data()
    x, y, query, weights, independent = [
        torch.as_tensor(item, dtype=torch.float64, device=device) for item in arrays
    ]
    identity = torch.eye(72, dtype=x.dtype, device=device)
    synchronize(device)
    setup_seconds = time.perf_counter() - started

    def calculate() -> torch.Tensor:
        coreg = weights @ weights.T + torch.diag(independent)
        joint = torch.kron(coreg, torch_rbf(x, x, 1.2, 0.75))
        joint = joint + 0.12 * identity
        cross = torch.kron(coreg, torch_rbf(query, x, 1.2, 0.75))
        alpha = torch.linalg.solve(joint, y.T.reshape(-1))
        return (cross @ alpha).reshape(3, 12).T.reshape(-1)

    return calculate, setup_seconds


def variational_data() -> tuple[np.ndarray, ...]:
    n, m, n_test = 64, 12, 20
    x = np.linspace(-2.0, 2.0, n)[:, None]
    y = np.sin(1.7 * x[:, 0]) + 0.05 * np.cos(np.arange(1, n + 1))
    inducing = np.linspace(-2.0, 2.0, m)[:, None]
    query = np.linspace(-1.9, 1.9, n_test)[:, None]
    mean = 0.25 * np.sin(0.4 * np.arange(1, m + 1))
    factor = np.diag(0.55 + 0.01 * np.arange(1, m + 1))
    factor[np.arange(1, m), np.arange(m - 1)] = 0.015
    return x, y, inducing, query, mean, factor


def numpy_variational_reference() -> np.ndarray:
    x, y, inducing, query, mean_parameter, factor = variational_data()
    k_uu = rbf(inducing, inducing, 1.3, 0.7)
    k_uu.flat[:: k_uu.shape[0] + 1] += 1.0e-10 * max(np.abs(k_uu).max(), 1.0)
    k_uf = rbf(inducing, x, 1.3, 0.7)
    k_us = rbf(inducing, query, 1.3, 0.7)
    a = np.linalg.solve(k_uu, k_uf)
    a_star = np.linalg.solve(k_uu, k_us)
    covariance = factor @ factor.T
    train_mean = a.T @ mean_parameter
    marginal = 1.3 - np.sum(a * k_uf, axis=0) + np.sum(a * (covariance @ a), axis=0)
    likelihood = np.sum(
        -0.5 * np.log(2.0 * np.pi * 0.2)
        - 0.5 * ((y - train_mean) ** 2 + marginal) / 0.2
    )
    trace = np.trace(np.linalg.solve(k_uu, covariance))
    quadratic = mean_parameter @ np.linalg.solve(k_uu, mean_parameter)
    divergence = 0.5 * (
        trace
        + quadratic
        - 12
        + np.linalg.slogdet(k_uu)[1]
        - 2.0 * np.log(np.diag(factor)).sum()
    )
    value = likelihood - divergence
    mean = a_star.T @ mean_parameter
    variance = 1.3 - np.sum(a_star * k_us, axis=0) + np.sum(
        a_star * (covariance @ a_star), axis=0
    )
    return np.concatenate(([value], mean, variance))


def prepare_torch_variational(
    device: torch.device,
) -> tuple[Callable[[], torch.Tensor], float]:
    started = time.perf_counter()
    arrays = [
        torch.as_tensor(item, dtype=torch.float64, device=device)
        for item in variational_data()
    ]
    x, y, inducing, query, mean_parameter, factor = arrays
    identity = torch.eye(12, dtype=x.dtype, device=device)
    synchronize(device)
    setup_seconds = time.perf_counter() - started

    def calculate() -> torch.Tensor:
        k_uu = torch_rbf(inducing, inducing, 1.3, 0.7)
        k_uu = k_uu + 1.3e-10 * identity
        k_uf = torch_rbf(inducing, x, 1.3, 0.7)
        k_us = torch_rbf(inducing, query, 1.3, 0.7)
        covariance = factor @ factor.T
        factor_kuu = torch.linalg.cholesky(k_uu)
        a = torch.cholesky_solve(k_uf, factor_kuu)
        a_star = torch.cholesky_solve(k_us, factor_kuu)
        train_mean = a.T @ mean_parameter
        marginal = 1.3 - (a * k_uf).sum(dim=0)
        marginal += (a * (covariance @ a)).sum(dim=0)
        likelihood = (
            -0.5 * math.log(2.0 * math.pi * 0.2)
            - 0.5 * ((y - train_mean) ** 2 + marginal) / 0.2
        ).sum()
        inverse_covariance = torch.cholesky_solve(covariance, factor_kuu)
        trace = torch.diagonal(inverse_covariance).sum()
        mean_solve = torch.cholesky_solve(
            mean_parameter[:, None], factor_kuu
        )[:, 0]
        divergence = 0.5 * (
            trace
            + mean_parameter @ mean_solve
            - 12
            + 2.0 * torch.log(torch.diagonal(factor_kuu)).sum()
            - 2.0 * torch.log(torch.diagonal(factor)).sum()
        )
        value = likelihood - divergence
        mean = a_star.T @ mean_parameter
        variance = 1.3 - (a_star * k_us).sum(dim=0)
        variance += (a_star * (covariance @ a_star)).sum(dim=0)
        return torch.cat((value.reshape(1), mean, variance))

    return calculate, setup_seconds


def prepare_gpytorch_logdet(
    device: torch.device,
) -> tuple[Callable[[], torch.Tensor], float]:
    import gpytorch

    started = time.perf_counter()
    x = torch.as_tensor(deterministic_points(64, 2), device=device)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(
        dtype=torch.float64, device=device
    )
    kernel.outputscale = SIGNAL
    kernel.base_kernel.lengthscale = SCALE
    diagonal = torch.full((64,), SHIFT, dtype=torch.float64, device=device)
    synchronize(device)
    setup_seconds = time.perf_counter() - started

    def calculate() -> torch.Tensor:
        # A fresh lazy wrapper prevents reuse of a cached decomposition while
        # retaining the fixed inputs and kernel prepared outside the timer.
        operator = kernel(x).add_diagonal(diagonal)
        torch.manual_seed(20260806)
        with (
            gpytorch.settings.max_cholesky_size(0),
            gpytorch.settings.num_trace_samples(64),
            gpytorch.settings.max_lanczos_quadrature_iterations(48),
        ):
            _, value = operator.inv_quad_logdet(logdet=True)
        return value.reshape(1)

    return calculate, setup_seconds


def prepare_gpytorch_predictive_variance(
    device: torch.device,
) -> tuple[Callable[[], torch.Tensor], float]:
    import gpytorch

    started = time.perf_counter()
    x = torch.as_tensor(deterministic_points(64, 2), device=device)
    query = torch.as_tensor(deterministic_queries(16, 2), device=device)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(
        dtype=torch.float64, device=device
    )
    kernel.outputscale = SIGNAL
    kernel.base_kernel.lengthscale = SCALE
    diagonal = torch.full((64,), SHIFT, dtype=torch.float64, device=device)
    synchronize(device)
    setup_seconds = time.perf_counter() - started

    def calculate() -> torch.Tensor:
        operator = kernel(x).add_diagonal(diagonal)
        cross = kernel(x, query).to_dense()
        torch.manual_seed(20260806)
        with (
            gpytorch.settings.max_cholesky_size(0),
            gpytorch.settings.max_root_decomposition_size(48),
        ):
            inverse_root = operator.root_inv_decomposition().root.to_dense()
        projected = inverse_root.T @ cross
        return SIGNAL - (projected * projected).sum(dim=0)

    return calculate, setup_seconds


def numpy_spectral_reference(workload: str) -> np.ndarray:
    x = deterministic_points(64, 2)
    matrix = rbf(x, x, SIGNAL, SCALE) + SHIFT * np.eye(64)
    if workload == "logdet":
        return np.array([np.linalg.slogdet(matrix)[1]])
    query = deterministic_queries(16, 2)
    cross = rbf(x, query, SIGNAL, SCALE)
    return SIGNAL - np.sum(cross * np.linalg.solve(matrix, cross), axis=0)


def numpy_reference(workload: str) -> np.ndarray:
    references: dict[str, Callable[[], np.ndarray]] = {
        "logdet": lambda: numpy_spectral_reference("logdet"),
        "predictive_variance": lambda: numpy_spectral_reference(
            "predictive_variance"
        ),
        "derivative": numpy_derivative_reference,
        "multi_output": numpy_multi_output_reference,
        "variational": numpy_variational_reference,
    }
    return references[workload]()


def benchmark_torch(
    workload: str, device: torch.device, repetitions: int
) -> tuple[str, float, float, float, int]:
    preparers: dict[
        str, Callable[[torch.device], tuple[Callable[[], torch.Tensor], float]]
    ] = {
        "logdet": prepare_gpytorch_logdet,
        "predictive_variance": prepare_gpytorch_predictive_variance,
        "derivative": prepare_torch_derivative,
        "multi_output": prepare_torch_multi_output,
        "variational": prepare_torch_variational,
    }
    backends = {
        "logdet": "gpytorch_slq",
        "predictive_variance": "gpytorch_love",
        "derivative": "pytorch_dense",
        "multi_output": "pytorch_dense",
        "variational": "pytorch_dense",
    }
    expected = numpy_reference(workload)
    function, setup_seconds = preparers[workload](device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        result = function()
    synchronize(device)
    actual = result.detach().cpu().numpy()
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError(f"{workload} {device} oracle contains a nonfinite value")
    error = float(
        np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1.0)
    )
    tolerance = 0.05 if workload in {"logdet", "predictive_variance"} else 2.0e-9
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(f"{workload} {device} oracle mismatch: {error:g}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repetitions):
            result = function()
    synchronize(device)
    seconds = (time.perf_counter() - start) / repetitions
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return backends[workload], seconds, error, setup_seconds, peak


def python_worker_record(
    workload: str, device_name: str, repetitions: int
) -> dict[str, object]:
    """Measure one reference workload in a fresh process."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(device_name)
    backend, seconds, error, setup_seconds, peak = benchmark_torch(
        workload, device, repetitions
    )
    return {
        "workload": workload,
        "device": device_name,
        "backend": backend,
        "status": "pass",
        "repetitions": repetitions,
        "warmups": 1,
        "seconds_per_call": seconds,
        "relative_error": error,
        "setup_seconds": setup_seconds,
        "peak_memory_bytes": peak,
    }


def validate_worker_record(
    record: Any, workload: str, device: str, repetitions: int
) -> dict[str, object]:
    """Validate the child JSON before it can enter the raw benchmark record."""
    if not isinstance(record, dict) or set(record) != set(WORKER_FIELDS):
        fields = sorted(record) if isinstance(record, dict) else type(record).__name__
        raise RuntimeError(f"invalid Python worker JSON fields: {fields}")
    expected_backend = {
        "logdet": "gpytorch_slq",
        "predictive_variance": "gpytorch_love",
        "derivative": "pytorch_dense",
        "multi_output": "pytorch_dense",
        "variational": "pytorch_dense",
    }[workload]
    expected = {
        "workload": workload,
        "device": device,
        "backend": expected_backend,
        "status": "pass",
        "repetitions": repetitions,
        "warmups": 1,
    }
    for field, value in expected.items():
        if record[field] != value:
            raise RuntimeError(
                f"invalid Python worker JSON {field}: {record[field]!r} != {value!r}"
            )
    for field in ("seconds_per_call", "relative_error", "setup_seconds"):
        try:
            value = float(record[field])
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"invalid Python worker JSON {field}") from error
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"nonfinite or negative Python worker {field}")
    try:
        peak = int(record["peak_memory_bytes"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("invalid Python worker JSON peak_memory_bytes") from error
    if peak <= 0:
        raise RuntimeError("nonpositive Python worker peak_memory_bytes")
    return record


def run_python_worker(
    root: Path,
    fortml: Path,
    workload: str,
    device: str,
    repetitions: int,
    cpu: int,
) -> dict[str, object]:
    """Run and schema-check one workload/device measurement child."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "record.json"
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
            "--repetitions",
            str(repetitions),
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
        try:
            record = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Python worker did not write valid JSON") from error
    return validate_worker_record(record, workload, device, repetitions)


def shape(workload: str) -> tuple[int, int, int, int, int]:
    return {
        "logdet": (64, 2, 1, 0, 0),
        "predictive_variance": (64, 2, 1, 16, 0),
        "derivative": (48, 1, 2, 24, 0),
        "multi_output": (24, 1, 3, 12, 0),
        "variational": (64, 1, 1, 20, 12),
    }[workload]


def plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    passed = [row for row in rows if row["status"] == "pass"]
    for row in passed:
        try:
            seconds = float(row["seconds_per_call"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("pass row contains an invalid timing") from error
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise RuntimeError("pass row contains a nonpositive or nonfinite timing")
    labels = list(
        dict.fromkeys(
            f"{row['backend']}-{row['device']}" for row in passed
        )
    )
    x = np.arange(len(WORKLOADS), dtype=np.float64)
    width = 0.8 / max(len(labels), 1)
    palette = ("#0072B2", "#E69F00", "#CC79A7", "#009E73")
    hatches = ("", "//", "xx", "..", "\\\\", "++", "oo")
    display_labels = {
        "fortml-cpu": "FortML, CPU",
        "gpytorch_slq-cpu": "GPyTorch SLQ, CPU",
        "gpytorch_slq-cuda": "GPyTorch SLQ, CUDA",
        "gpytorch_love-cpu": "GPyTorch LOVE, CPU",
        "gpytorch_love-cuda": "GPyTorch LOVE, CUDA",
        "pytorch_dense-cpu": "PyTorch dense, CPU",
        "pytorch_dense-cuda": "PyTorch dense, CUDA",
    }
    figure, axis = plt.subplots(figsize=(10.0, 6.2))
    for index, label in enumerate(labels):
        values = []
        for workload in WORKLOADS:
            match = [
                row
                for row in passed
                if row["workload"] == workload
                and f"{row['backend']}-{row['device']}" == label
            ]
            values.append(
                1000.0 * float(match[0]["seconds_per_call"])
                if match
                else np.nan
            )
        axis.bar(
            x + (index - (len(labels) - 1) / 2) * width,
            values,
            width,
            label=display_labels.get(label, label),
            color=palette[index % len(palette)],
            edgecolor="#202020",
            linewidth=0.7,
            hatch=hatches[index % len(hatches)],
        )
    axis.set_yscale("log")
    axis.set_ylabel("milliseconds per complete call (log scale)")
    axis.set_xticks(x, [name.replace("_", "\n") for name in WORKLOADS])
    axis.set_title(
        "GP feature workloads: matched timed phases\n"
        "float64, one CPU thread; deterministic input generation excluded"
    )
    axis.legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.subplots_adjust(bottom=0.29, left=0.10, right=0.98, top=0.84)
    figure.text(
        0.5,
        0.01,
        "FortML CUDA rows are unsupported; missing bars are explicit in the CSV.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/gp_features.csv"))
    parser.add_argument(
        "--plot", type=Path, default=Path("results/gp_features.png")
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--compiler", default=os.environ.get("FC", "gfortran"))
    parser.add_argument("--flags", default="-O3 -funroll-loops")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--worker", choices=WORKLOADS, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--worker-output", type=Path, default=None)
    parser.add_argument("--cpu", type=int, default=None)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("repetitions must be positive")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    cpu = args.cpu if args.cpu is not None else allowed_cpus[0]
    if cpu not in allowed_cpus:
        raise SystemExit(f"CPU {cpu} is outside this process's affinity mask")
    os.sched_setaffinity(0, {cpu})
    fortml = args.fortml.resolve()
    root = Path(__file__).resolve().parents[1]
    if args.worker is not None:
        if args.worker_output is None:
            raise SystemExit("--worker-output is required with --worker")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA worker requested but torch.cuda.is_available() is false")
        record = python_worker_record(args.worker, args.device, args.repetitions)
        validate_worker_record(record, args.worker, args.device, args.repetitions)
        args.worker_output.write_text(json.dumps(record, allow_nan=False))
        return

    app_path = fortml / "app" / "fortml_bench_gp_features.f90"
    lanczos_path = fortml / "src" / "gp" / "fortml_lanczos.f90"
    build_seconds = build_fortml(fortml, args.compiler, args.flags)
    gpu, driver = gpu_metadata()
    metadata = versions()
    metadata.update(
        {
            "threads": 1,
            "cpu_affinity": cpu,
            "compiler": args.compiler,
            "compiler_version": compiler_version(args.compiler),
            "flags": args.flags,
            "cpu_model": cpu_model(),
            "gpu_model": gpu,
            "driver_version": driver,
            "os": platform.platform(),
            "fortml_revision": git_revision(fortml),
            "fortnum_revision": git_revision(fortml.parent / "fortnum"),
            "benchmark_revision": git_revision(root),
            "fortml_diff_sha256": tracked_diff_sha256(fortml),
            "driver_sha256": sha256(Path(__file__).resolve()),
            "app_sha256": sha256(app_path),
            "lanczos_source_sha256": sha256(lanczos_path),
        }
    )
    rows: list[dict[str, object]] = []
    devices = ["cpu"]
    cuda_enabled = torch.cuda.is_available() and not args.cpu_only
    if cuda_enabled:
        devices.append("cuda")
    for workload in WORKLOADS:
        expected = numpy_reference(workload)
        result, peak, external_error = run_fortml(
            fortml, workload, args.repetitions, args.compiler, expected
        )
        rows.append(
            {
                **result,
                "relative_error": external_error,
                "backend": "fortml",
                "device": "cpu",
                "status": "pass",
                "warmups": 1,
                "setup_seconds": "",
                "build_seconds": build_seconds,
                "peak_memory_bytes": peak,
                "executable_bytes": executable_size(fortml),
                "oracle": "independent_numpy_full_output",
                "notes": (
                    "host public API; internal check error="
                    + result["relative_error"]
                ),
                **metadata,
            }
        )
        for device in devices:
            worker = run_python_worker(
                root, fortml, workload, device, args.repetitions, cpu
            )
            n, d, p, n_test, n_inducing = shape(workload)
            memory_note = (
                "isolated-process max RSS"
                if device == "cpu"
                else "reset CUDA allocation peak"
            )
            rows.append(
                {
                    "workload": workload,
                    "n_samples": n,
                    "n_features": d,
                    "n_outputs": p,
                    "n_test": n_test,
                    "n_inducing": n_inducing,
                    "repetitions": worker["repetitions"],
                    "seconds_per_call": worker["seconds_per_call"],
                    "relative_error": worker["relative_error"],
                    "backend": worker["backend"],
                    "device": worker["device"],
                    "status": worker["status"],
                    "warmups": worker["warmups"],
                    "setup_seconds": worker["setup_seconds"],
                    "build_seconds": 0.0,
                    "peak_memory_bytes": worker["peak_memory_bytes"],
                    "executable_bytes": "not_applicable",
                    "oracle": "independent_numpy_dense_linear_algebra",
                    "notes": (
                        "deterministic tensors and immutable parameters prepared "
                        f"before timing; {memory_note}"
                    ),
                    **metadata,
                    "compiler": "python",
                    "compiler_version": "",
                    "flags": "torch.set_num_threads(1)",
                }
            )
        if cuda_enabled:
            n, d, p, n_test, n_inducing = shape(workload)
            rows.append(
                {
                    "workload": workload,
                    "n_samples": n,
                    "n_features": d,
                    "n_outputs": p,
                    "n_test": n_test,
                    "n_inducing": n_inducing,
                    "repetitions": args.repetitions,
                    "seconds_per_call": "",
                    "relative_error": "",
                    "backend": "fortml",
                    "device": "cuda",
                    "status": "unsupported",
                    "warmups": 0,
                    "setup_seconds": "",
                    "build_seconds": build_seconds,
                    "peak_memory_bytes": "",
                    "executable_bytes": executable_size(fortml),
                    "oracle": "not_applicable",
                    "notes": "FortML workload has no device-resident public path",
                    **metadata,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in CSV_FIELDS} for row in rows
        )
    plot(rows, args.plot)
    print(args.output)
    print(args.plot)


if __name__ == "__main__":
    main()
