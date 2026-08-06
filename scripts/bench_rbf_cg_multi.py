from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.bench_rbf_mvm import (
        DIAGONAL_SHIFT,
        LENGTHSCALE,
        VARIANCE,
        prepare_keops_runtime,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.bench_rbf_mvm import (
        DIAGONAL_SHIFT,
        LENGTHSCALE,
        VARIANCE,
        prepare_keops_runtime,
    )

DEFAULT_N = 2048
DEFAULT_D = 8
DEFAULT_RHS = 4
DEFAULT_REPETITIONS = 3
DEFAULT_TOLERANCE = 1.0e-8
DEFAULT_MAX_ITERATIONS = 500
DEFAULT_ORACLE_N = 256


def make_inputs(
    n_samples: int,
    n_features: int,
    n_rhs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(1, n_samples + 1, dtype=dtype, device=device)
    points = torch.empty((n_samples, n_features), dtype=dtype, device=device)
    for feature in range(1, n_features + 1):
        points[:, feature - 1] = (
            torch.sin(0.013 * (indices + 3 * feature))
            + 0.1 * torch.cos(0.017 * (indices * feature))
        )
    rhs = torch.empty((n_samples, n_rhs), dtype=dtype, device=device)
    for column in range(1, n_rhs + 1):
        rhs[:, column - 1] = (
            torch.sin(0.021 * (indices + 2 * column))
            + 0.3 * torch.cos(0.007 * (2 * indices + column))
        )
    return points, rhs


def dense_operator(points: torch.Tensor):
    distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(dim=2)
    matrix = DIAGONAL_SHIFT * torch.eye(
        points.shape[0], dtype=points.dtype, device=points.device
    )
    matrix = matrix + VARIANCE * torch.exp(-0.5 * distances / LENGTHSCALE**2)

    def matmat(matrix_rhs: torch.Tensor) -> torch.Tensor:
        return matrix @ matrix_rhs

    return matmat


def keops_operator(points: torch.Tensor):
    from pykeops.torch import LazyTensor

    points_i = LazyTensor(points[:, None, :])
    points_j = LazyTensor(points[None, :, :])
    distances = ((points_i - points_j) ** 2).sum(-1)

    def matmat(matrix_rhs: torch.Tensor) -> torch.Tensor:
        rhs_j = LazyTensor(matrix_rhs[None, :, :])
        products = VARIANCE * (-0.5 * distances / LENGTHSCALE**2).exp() * rhs_j
        return DIAGONAL_SHIFT * matrix_rhs + products.sum(dim=1).reshape(
            points.shape[0], matrix_rhs.shape[1]
        )

    return matmat


def gpytorch_keops_operator(points: torch.Tensor):
    import gpytorch
    from gpytorch.settings import max_cholesky_size, use_keops

    with torch.no_grad(), use_keops(True), max_cholesky_size(0):
        kernel = gpytorch.kernels.keops.RBFKernel().to(
            device=points.device, dtype=points.dtype
        )
        kernel.lengthscale = LENGTHSCALE
        operator = kernel(points, points)

    def matmat(matrix_rhs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), use_keops(True), max_cholesky_size(0):
            return DIAGONAL_SHIFT * matrix_rhs + VARIANCE * operator.matmul(matrix_rhs)

    return matmat


def cg_multi_solve(
    matmat,
    right_hand_side: torch.Tensor,
    tolerance: float,
    max_iterations: int,
):
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side - matmat(solution)
    direction = residual.clone()
    rhs_norm = torch.linalg.vector_norm(right_hand_side, dim=0)
    target = tolerance * torch.maximum(rhs_norm, torch.ones_like(rhs_norm))
    residual_norm = torch.linalg.vector_norm(residual, dim=0)
    active = residual_norm > target
    direction[:, ~active] = 0.0
    rho = (residual * residual).sum(dim=0)
    iterations = torch.zeros(
        right_hand_side.shape[1], dtype=torch.int64, device=right_hand_side.device
    )

    while bool(active.any()):
        operator_direction = matmat(direction)
        denominator = (direction * operator_direction).sum(dim=0)
        if bool((denominator[active] <= 0.0).any()) or not bool(
            torch.isfinite(denominator[active]).all()
        ):
            raise RuntimeError("multi-RHS CG breakdown")
        step = torch.zeros_like(rho)
        step[active] = rho[active] / denominator[active]
        solution = solution + direction * step[None, :]
        residual = residual - operator_direction * step[None, :]
        iterations[active] += 1
        residual_norm = torch.linalg.vector_norm(residual, dim=0)
        candidate = active & ((residual_norm <= target) | (iterations >= max_iterations))
        if bool(candidate.any()):
            residual = right_hand_side - matmat(solution)
            residual_norm = torch.linalg.vector_norm(residual, dim=0)
            converged = candidate & (residual_norm <= target)
            active[converged] = False
            direction[:, converged] = 0.0
            exhausted = candidate & (iterations >= max_iterations) & ~converged
            active[exhausted] = False
            direction[:, exhausted] = 0.0
        if bool(active.any()):
            next_rho = (residual * residual).sum(dim=0)
            if bool((next_rho[active] <= 0.0).any()) or not bool(
                torch.isfinite(next_rho[active]).all()
            ):
                raise RuntimeError("multi-RHS CG breakdown")
            beta = torch.zeros_like(rho)
            beta[active] = next_rho[active] / rho[active]
            direction[:, active] = residual[:, active] + direction[:, active] * beta[active]
            rho[active] = next_rho[active]

    residual = right_hand_side - matmat(solution)
    residual_norm = torch.linalg.vector_norm(residual, dim=0)
    return solution, iterations.cpu().numpy(), residual_norm, target


def independent_matmat(points: np.ndarray, matrix_rhs: np.ndarray) -> np.ndarray:
    result = DIAGONAL_SHIFT * matrix_rhs.copy()
    block_size = 256
    for first in range(0, points.shape[0], block_size):
        last = min(first + block_size, points.shape[0])
        distances = ((points[first:last, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        weights = VARIANCE * np.exp(-0.5 * distances / LENGTHSCALE**2)
        result[first:last] += weights @ matrix_rhs
    return result


def module_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "unavailable"
    module = __import__(name)
    return str(getattr(module, "__version__", "unknown"))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_backend(
    factory,
    points: torch.Tensor,
    right_hand_side: torch.Tensor,
    device: torch.device,
    repetitions: int,
    tolerance: float,
    max_iterations: int,
):
    synchronize(device)
    setup_start = time.perf_counter()
    matmat = factory(points)
    solution, iterations, _, _ = cg_multi_solve(
        matmat, right_hand_side, tolerance, max_iterations
    )
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start

    synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        solution, iterations, residual_norm, target = cg_multi_solve(
            matmat, right_hand_side, tolerance, max_iterations
        )
    synchronize(device)
    seconds_per_solve = (time.perf_counter() - start) / repetitions
    return (
        solution.detach().cpu().numpy(),
        setup_seconds,
        seconds_per_solve,
        iterations,
        residual_norm.detach().cpu().numpy(),
        target.detach().cpu().numpy(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--rhs", type=int, default=DEFAULT_RHS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--oracle-n", type=int, default=DEFAULT_ORACLE_N)
    parser.add_argument("--output", type=Path, default=Path("results/rbf_cg_multi.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.n, args.d, args.rhs, args.repetitions, args.max_iterations) < 1:
        raise SystemExit("n, d, rhs, repetitions, and max-iterations must be positive")
    if args.tolerance <= 0.0 or args.oracle_n < 1:
        raise SystemExit("tolerance and oracle-n must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    prepare_keops_runtime()
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    points_cpu, rhs_cpu = make_inputs(args.n, args.d, args.rhs, dtype, torch.device("cpu"))
    points = points_cpu.to(device)
    right_hand_side = rhs_cpu.to(device)
    points_np = points_cpu.numpy()
    rhs_np = rhs_cpu.numpy()
    independent_residual_limit = 5.0e-7 * max(1.0, args.n / 2048.0)
    expected_solution = None
    if args.n <= args.oracle_n:
        distances = ((points_np[:, None, :] - points_np[None, :, :]) ** 2).sum(axis=2)
        matrix = DIAGONAL_SHIFT * np.eye(args.n) + VARIANCE * np.exp(
            -0.5 * distances / LENGTHSCALE**2
        )
        expected_solution = np.linalg.solve(matrix, rhs_np)

    backends = {
        "pytorch_dense": dense_operator,
        "keops": keops_operator,
        "gpytorch_keops": gpytorch_keops_operator,
    }
    rows = []
    for backend, factory in backends.items():
        if args.device == "cuda":
            torch.cuda.empty_cache()
        try:
            (
                solution,
                setup_seconds,
                seconds_per_solve,
                iterations,
                reported_residual,
                target,
            ) = run_backend(
                factory,
                points,
                right_hand_side,
                device,
                args.repetitions,
                args.tolerance,
                args.max_iterations,
            )
            independent_residual = independent_matmat(points_np, solution) - rhs_np
            relative_residual = np.linalg.norm(independent_residual) / max(
                np.linalg.norm(rhs_np), 1.0
            )
            if relative_residual > independent_residual_limit:
                raise RuntimeError(
                    "independent residual check failed: "
                    f"{relative_residual:g} > {independent_residual_limit:g}"
                )
            if expected_solution is None:
                solution_error = math.nan
            else:
                solution_error = float(
                    np.max(np.abs(solution - expected_solution))
                    / max(1.0, np.max(np.abs(expected_solution)))
                )
                if solution_error > 1.0e-5:
                    raise RuntimeError(f"dense oracle check failed: {solution_error:g}")
            status = "pass"
        except (ImportError, MemoryError, OSError, RuntimeError) as exc:
            message = str(exc).replace("\n", " ")
            if "out of memory" in message.lower() or "memory" in message.lower():
                status = "oom"
            else:
                if isinstance(exc, RuntimeError):
                    raise
                status = "unsupported"
            setup_seconds = math.nan
            seconds_per_solve = math.nan
            iterations = []
            reported_residual = np.full(args.rhs, math.nan)
            target = np.full(args.rhs, math.nan)
            relative_residual = math.nan
            solution_error = math.nan
        rows.append(
            {
                "backend": backend,
                "device": args.device,
                "residency": "resident_inputs",
                "n_samples": args.n,
                "n_features": args.d,
                "n_rhs": args.rhs,
                "dtype": args.dtype,
                "threads": args.threads,
                "repetitions": args.repetitions,
                "tolerance": args.tolerance,
                "max_iterations": args.max_iterations,
                "iterations_max": int(max(iterations)) if len(iterations) else "",
                "setup_seconds": setup_seconds,
                "seconds_per_solve": seconds_per_solve,
                "reported_residual_norm_max": float(np.max(reported_residual)),
                "target_residual_norm_max": float(np.max(target)),
                "independent_relative_residual": relative_residual,
                "independent_residual_limit": independent_residual_limit,
                "dense_solution_relative_error": solution_error,
                "status": status,
                "torch_version": torch.__version__,
                "gpytorch_version": module_version("gpytorch"),
                "pykeops_version": module_version("pykeops"),
                "cuda_device": torch.cuda.get_device_name(0)
                if args.device == "cuda"
                else "cpu",
                "correctness_oracle": "blocked_numpy_matmat_and_small_dense_solve",
            }
        )
        if args.device == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    sys.exit(main())
