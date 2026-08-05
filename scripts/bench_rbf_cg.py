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
        make_inputs,
        prepare_keops_runtime,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.bench_rbf_mvm import (
        DIAGONAL_SHIFT,
        LENGTHSCALE,
        VARIANCE,
        make_inputs,
        prepare_keops_runtime,
    )

DEFAULT_N = 2048
DEFAULT_D = 8
DEFAULT_REPETITIONS = 3
DEFAULT_TOLERANCE = 1.0e-8
DEFAULT_MAX_ITERATIONS = 500
DEFAULT_ORACLE_N = 256


def dense_operator(points: torch.Tensor):
    distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(dim=2)
    matrix = DIAGONAL_SHIFT * torch.eye(
        points.shape[0], dtype=points.dtype, device=points.device
    )
    matrix = matrix + VARIANCE * torch.exp(-0.5 * distances / LENGTHSCALE**2)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return matrix @ vector

    return matvec


def keops_operator(points: torch.Tensor):
    from pykeops.torch import LazyTensor

    points_i = LazyTensor(points[:, None, :])
    points_j = LazyTensor(points[None, :, :])

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        vector_j = LazyTensor(vector[None, :, None])
        distances = ((points_i - points_j) ** 2).sum(-1)
        products = VARIANCE * (-0.5 * distances / LENGTHSCALE**2).exp() * vector_j
        return DIAGONAL_SHIFT * vector + products.sum(dim=1).reshape(-1)

    return matvec


def gpytorch_keops_operator(points: torch.Tensor):
    import gpytorch
    from gpytorch.settings import max_cholesky_size, use_keops

    with torch.no_grad(), use_keops(True), max_cholesky_size(0):
        kernel = gpytorch.kernels.keops.RBFKernel().to(
            device=points.device, dtype=points.dtype
        )
        kernel.lengthscale = LENGTHSCALE
        operator = kernel(points, points)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), use_keops(True), max_cholesky_size(0):
            return DIAGONAL_SHIFT * vector + VARIANCE * operator.matmul(
                vector[:, None]
            ).reshape(-1)

    return matvec


def cg_solve(
    matvec,
    right_hand_side: torch.Tensor,
    tolerance: float,
    max_iterations: int,
):
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side - matvec(solution)
    direction = residual.clone()
    rho = torch.dot(residual, residual)
    right_hand_side_norm = torch.linalg.vector_norm(right_hand_side).item()
    target = tolerance * max(right_hand_side_norm, 1.0)
    residual_norm = torch.linalg.vector_norm(residual).item()
    if residual_norm <= target:
        return solution, 0, residual_norm, target
    for iteration in range(1, max_iterations + 1):
        operator_direction = matvec(direction)
        denominator = torch.dot(direction, operator_direction)
        if not bool(torch.isfinite(denominator)) or denominator.item() <= 0.0:
            raise RuntimeError("CG breakdown")
        step = rho / denominator
        solution = solution + step * direction
        residual = residual - step * operator_direction
        residual_norm = torch.linalg.vector_norm(residual).item()
        if residual_norm <= target:
            residual = right_hand_side - matvec(solution)
            residual_norm = torch.linalg.vector_norm(residual).item()
            if residual_norm <= target:
                return solution, iteration, residual_norm, target
            direction = residual.clone()
            rho = torch.dot(residual, residual)
            continue
        next_rho = torch.dot(residual, residual)
        if not bool(torch.isfinite(next_rho)) or next_rho.item() <= 0.0:
            raise RuntimeError("CG breakdown")
        direction = residual + (next_rho / rho) * direction
        rho = next_rho
    residual = right_hand_side - matvec(solution)
    residual_norm = torch.linalg.vector_norm(residual).item()
    return solution, max_iterations, residual_norm, target


def independent_matvec(points: np.ndarray, vector: np.ndarray) -> np.ndarray:
    result = DIAGONAL_SHIFT * vector.copy()
    block_size = 256
    for first in range(0, points.shape[0], block_size):
        last = min(first + block_size, points.shape[0])
        distances = (
            (points[first:last, None, :] - points[None, :, :]) ** 2
        ).sum(axis=2)
        weights = VARIANCE * np.exp(-0.5 * distances / LENGTHSCALE**2)
        result[first:last] += weights @ vector
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
    matvec = factory(points)
    solution, iterations, _, _ = cg_solve(
        matvec, right_hand_side, tolerance, max_iterations
    )
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start

    synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        solution, iterations, residual_norm, target = cg_solve(
            matvec, right_hand_side, tolerance, max_iterations
        )
    synchronize(device)
    seconds_per_solve = (time.perf_counter() - start) / repetitions
    return (
        solution.detach().cpu().numpy(),
        setup_seconds,
        seconds_per_solve,
        iterations,
        residual_norm,
        target,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--oracle-n", type=int, default=DEFAULT_ORACLE_N)
    parser.add_argument("--output", type=Path, default=Path("results/rbf_cg.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.n, args.d, args.repetitions, args.max_iterations) < 1:
        raise SystemExit("n, d, repetitions, and max-iterations must be positive")
    if args.tolerance <= 0.0 or args.oracle_n < 1:
        raise SystemExit("tolerance and oracle-n must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    prepare_keops_runtime()
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    points_cpu, rhs_cpu = make_inputs(args.n, args.d, dtype, torch.device("cpu"))
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
            independent_residual = independent_matvec(points_np, solution) - rhs_np
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
                    raise RuntimeError(
                        f"dense oracle check failed: {solution_error:g}"
                    )
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
            iterations = ""
            reported_residual = math.nan
            target = math.nan
            relative_residual = math.nan
            solution_error = math.nan
        rows.append(
            {
                "backend": backend,
                "device": args.device,
                "residency": "resident_inputs",
                "n_samples": args.n,
                "n_features": args.d,
                "dtype": args.dtype,
                "threads": args.threads,
                "repetitions": args.repetitions,
                "tolerance": args.tolerance,
                "max_iterations": args.max_iterations,
                "iterations": iterations,
                "setup_seconds": setup_seconds,
                "seconds_per_solve": seconds_per_solve,
                "reported_residual_norm": reported_residual,
                "target_residual_norm": target,
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
                "correctness_oracle": "blocked_numpy_matvec_and_small_dense_solve",
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
