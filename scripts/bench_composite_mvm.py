from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.bench_rbf_mvm import (
        make_inputs,
        prepare_keops_runtime,
        synchronize,
        timed,
        version,
    )
except ImportError:
    from bench_rbf_mvm import (  # type: ignore
        make_inputs,
        prepare_keops_runtime,
        synchronize,
        timed,
        version,
    )

VARIANCE = 1.4
LENGTHSCALE = 0.7
CONSTANT_VARIANCE = 0.2
DIAGONAL_SHIFT = 0.08


def oracle(points: torch.Tensor, vector: torch.Tensor) -> np.ndarray:
    x = points.detach().cpu().numpy()
    v = vector.detach().cpu().numpy()
    result = (DIAGONAL_SHIFT * v) + CONSTANT_VARIANCE * np.sum(v)
    block = 256
    for first in range(0, x.shape[0], block):
        last = min(first + block, x.shape[0])
        distances = ((x[first:last, None, :] - x[None, :, :]) ** 2).sum(axis=2)
        weights = VARIANCE * np.exp(-0.5 * distances / (LENGTHSCALE**2))
        result[first:last] += weights @ v
    return result


def dense_torch(points: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(dim=2)
    kernel = VARIANCE * torch.exp(-0.5 * distances / LENGTHSCALE**2)
    return (
        DIAGONAL_SHIFT * vector
        + CONSTANT_VARIANCE * vector.sum()
        + kernel @ vector
    )


def keops_torch(points: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    from pykeops.torch import LazyTensor

    points_i = LazyTensor(points[:, None, :])
    points_j = LazyTensor(points[None, :, :])
    vector_j = LazyTensor(vector[None, :, None])
    distances = ((points_i - points_j) ** 2).sum(-1)
    products = VARIANCE * (-0.5 * distances / LENGTHSCALE**2).exp() * vector_j
    return (
        DIAGONAL_SHIFT * vector
        + CONSTANT_VARIANCE * vector.sum()
        + products.sum(dim=1).reshape(-1)
    )


def gpytorch_keops(points: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    import gpytorch
    from gpytorch.settings import max_cholesky_size, use_keops

    kernel = gpytorch.kernels.keops.RBFKernel().to(
        device=points.device, dtype=points.dtype
    )
    kernel.lengthscale = LENGTHSCALE
    with torch.no_grad(), use_keops(True), max_cholesky_size(0):
        operator = kernel(points, points)
        rbf_result = operator.matmul(vector[:, None]).reshape(-1)
    return (
        DIAGONAL_SHIFT * vector
        + CONSTANT_VARIANCE * vector.sum()
        + VARIANCE * rbf_result
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--output", type=Path, default=Path("results/composite_mvm.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 1 or args.d < 1 or args.repetitions < 1:
        raise SystemExit("n, d, and repetitions must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    prepare_keops_runtime()
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    points_cpu, vector_cpu = make_inputs(args.n, args.d, dtype, torch.device("cpu"))
    expected = oracle(points_cpu, vector_cpu)
    backends = {
        "pytorch_dense": dense_torch,
        "keops": keops_torch,
        "gpytorch_keops": gpytorch_keops,
    }
    rows = []
    for backend, function in backends.items():
        residencies = ("resident", "transfer") if args.device == "cuda" else ("resident",)
        for residency in residencies:
            if args.device == "cuda":
                torch.cuda.empty_cache()
            try:
                result, setup, elapsed = timed(
                    function,
                    points_cpu,
                    vector_cpu,
                    device,
                    residency,
                    args.repetitions,
                )
                error = float(np.max(np.abs(result - expected)))
                scale = max(1.0, float(np.max(np.abs(expected))))
                relative_error = error / scale
                limit = 5.0e-5 if args.dtype == "float32" else 1.0e-6
                if relative_error > limit:
                    raise RuntimeError(
                        f"independent oracle mismatch: {relative_error:g}"
                    )
                status = "pass"
            except (MemoryError, ImportError, OSError, RuntimeError) as exc:
                message = str(exc).replace("\n", " ")
                if "out of memory" in message.lower() or "memory" in message.lower():
                    status = "oom"
                elif isinstance(exc, RuntimeError):
                    raise
                else:
                    status = "unsupported"
                setup = math.nan
                elapsed = math.nan
                relative_error = math.nan
            rows.append(
                {
                    "workload": "rbf_plus_constant",
                    "backend": backend,
                    "device": args.device,
                    "residency": residency,
                    "n_samples": args.n,
                    "n_features": args.d,
                    "dtype": args.dtype,
                    "threads": args.threads,
                    "repetitions": args.repetitions,
                    "setup_seconds": setup,
                    "seconds_per_mvm": elapsed,
                    "relative_error": relative_error,
                    "status": status,
                    "torch_version": torch.__version__,
                    "gpytorch_version": version("gpytorch"),
                    "pykeops_version": version("pykeops"),
                    "cuda_device": torch.cuda.get_device_name(0)
                    if args.device == "cuda"
                    else "cpu",
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
    main()
