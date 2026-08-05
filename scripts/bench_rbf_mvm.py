from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

VARIANCE = 1.4
LENGTHSCALE = 0.7
DIAGONAL_SHIFT = 0.08
DEFAULT_N = 2048
DEFAULT_D = 8
DEFAULT_REPETITIONS = 12


def prepare_keops_runtime() -> None:
    """Make versioned CUDA runtime libraries linkable by KeOps' JIT build."""
    candidates = []
    for search_path in sys.path:
        candidate_root = Path(search_path) / "nvidia"
        candidates.extend(candidate_root.glob("cu*/lib/libnvrtc.so.*"))
    candidates.extend(Path("/opt/cuda").glob("**/libnvrtc.so.*"))
    if not candidates:
        return
    library = candidates[0].resolve()
    directory = library.parent
    unversioned = directory / "libnvrtc.so"
    try:
        if not unversioned.exists():
            unversioned.symlink_to(library.name)
    except OSError:
        return
    library_path = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if str(directory) not in library_path:
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [str(directory)] + [entry for entry in library_path if entry]
        )


def make_inputs(n: int, d: int, dtype: torch.dtype, device: torch.device):
    indices = torch.arange(1, n + 1, dtype=dtype, device=device)
    features = torch.arange(1, d + 1, dtype=dtype, device=device)
    points = torch.sin(0.013 * (indices[:, None] + 3 * features[None, :]))
    points = points + 0.1 * torch.cos(0.017 * indices[:, None] * features[None, :])
    vector = torch.sin(0.021 * indices)
    vector = vector + 0.3 * torch.cos(0.007 * (2 * indices + 1))
    return points, vector


def oracle(points: torch.Tensor, vector: torch.Tensor) -> np.ndarray:
    x = points.detach().cpu().numpy()
    v = vector.detach().cpu().numpy()
    result = DIAGONAL_SHIFT * v.copy()
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
    return DIAGONAL_SHIFT * vector + kernel @ vector


def keops_torch(points: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    from pykeops.torch import LazyTensor

    points_i = LazyTensor(points[:, None, :])
    points_j = LazyTensor(points[None, :, :])
    vector_j = LazyTensor(vector[None, :, None])
    distances = ((points_i - points_j) ** 2).sum(-1)
    products = VARIANCE * (-0.5 * distances / LENGTHSCALE**2).exp() * vector_j
    return DIAGONAL_SHIFT * vector + products.sum(dim=1).reshape(-1)


def gpytorch_keops(points: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    import gpytorch
    from gpytorch.settings import max_cholesky_size, use_keops

    kernel = gpytorch.kernels.keops.RBFKernel().to(
        device=points.device, dtype=points.dtype
    )
    kernel.lengthscale = LENGTHSCALE
    with torch.no_grad(), use_keops(True), max_cholesky_size(0):
        operator = kernel(points, points)
        result = operator.matmul(vector[:, None]).reshape(-1)
    return DIAGONAL_SHIFT * vector + VARIANCE * result


def version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "unavailable"
    module = __import__(name)
    return str(getattr(module, "__version__", "unknown"))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(
    fn,
    points_cpu: torch.Tensor,
    vector_cpu: torch.Tensor,
    device: torch.device,
    residency: str,
    repetitions: int,
):
    if residency == "resident":
        points = points_cpu.to(device)
        vector = vector_cpu.to(device)
        synchronize(device)
        start_setup = time.perf_counter()
        result = fn(points, vector)
        synchronize(device)
        setup = time.perf_counter() - start_setup
        start = time.perf_counter()
        for _ in range(repetitions):
            result = fn(points, vector)
        synchronize(device)
        elapsed = (time.perf_counter() - start) / repetitions
    else:
        synchronize(device)
        start_setup = time.perf_counter()
        points = points_cpu.to(device)
        vector = vector_cpu.to(device)
        result = fn(points, vector)
        synchronize(device)
        result = result.cpu()
        setup = time.perf_counter() - start_setup
        start = time.perf_counter()
        for _ in range(repetitions):
            points = points_cpu.to(device)
            vector = vector_cpu.to(device)
            result = fn(points, vector).cpu()
        synchronize(device)
        elapsed = (time.perf_counter() - start) / repetitions
    return result.detach().cpu().numpy(), setup, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=Path("results/rbf_mvm.csv"))
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
        for residency in ("resident", "transfer") if args.device == "cuda" else ("resident",):
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
                if relative_error > (5.0e-5 if args.dtype == "float32" else 1.0e-6):
                    raise RuntimeError(f"independent oracle mismatch: {relative_error:g}")
                status = "pass"
            except (MemoryError, ImportError, OSError, RuntimeError) as exc:
                message = str(exc).replace("\n", " ")
                if "out of memory" in message.lower() or "memory" in message.lower():
                    status = "oom"
                else:
                    if isinstance(exc, RuntimeError):
                        raise
                    status = "unsupported"
                setup = math.nan
                elapsed = math.nan
                relative_error = math.nan
            rows.append(
                {
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
