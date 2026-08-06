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

DIAGONAL_SHIFT = 0.08
DEFAULT_N = 4096
DEFAULT_RADIUS = 8
DEFAULT_RHS = 4
DEFAULT_REPETITIONS = 40


def prepare_keops_runtime() -> None:
    candidates = []
    for search_path in sys.path:
        candidates.extend((Path(search_path) / "nvidia").glob("cu*/lib/libnvrtc.so.*"))
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


def make_inputs(n: int, rhs: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(1, n + 1, dtype=dtype)
    columns = torch.arange(1, rhs + 1, dtype=dtype)
    inputs = torch.sin(0.013 * (indices[:, None] + 3 * columns[None, :]))
    inputs = inputs + 0.1 * torch.cos(0.017 * (2 * indices[:, None] + columns[None, :]))
    return indices[:, None], inputs


def wendland_c2(distance: np.ndarray) -> np.ndarray:
    remainder = np.maximum(0.0, 1.0 - distance)
    return remainder**4 * (4.0 * distance + 1.0)


def oracle(n: int, radius: int, inputs: torch.Tensor) -> np.ndarray:
    values = inputs.numpy()
    result = np.zeros_like(values)
    for row in range(n):
        first = max(0, row - radius)
        last = min(n, row + radius + 1)
        distance = np.abs(np.arange(first, last) - row)/radius
        weights = wendland_c2(distance)
        weights[row - first] += DIAGONAL_SHIFT
        result[row] = weights @ values[first:last]
    return result


def dense_torch(indices: torch.Tensor, inputs: torch.Tensor, radius: int) -> torch.Tensor:
    distance = (indices[:, None, :] - indices[None, :, :]).abs()/radius
    kernel = torch.relu(1.0 - distance)**4 * (4.0 * distance + 1.0)
    kernel = kernel + DIAGONAL_SHIFT*torch.eye(
        indices.shape[0], dtype=indices.dtype, device=indices.device
    )[:, :, None]
    return (kernel*inputs[None, :, :]).sum(dim=1)


def keops_torch(indices: torch.Tensor, inputs: torch.Tensor, radius: int) -> torch.Tensor:
    from pykeops.torch import LazyTensor

    indices_i = LazyTensor(indices[:, None, :])
    indices_j = LazyTensor(indices[None, :, :])
    inputs_j = LazyTensor(inputs[None, :, :])
    distance = ((indices_i - indices_j)**2).sum(-1).sqrt()/radius
    kernel = (1.0 - distance).relu()**4 * (4.0*distance + 1.0)
    result = (kernel*inputs_j).sum(dim=1).reshape(inputs.shape)
    return result + DIAGONAL_SHIFT*inputs


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, indices_cpu, inputs_cpu, device, radius, residency, repetitions):
    if residency == "resident":
        indices = indices_cpu.to(device)
        inputs = inputs_cpu.to(device)
        synchronize(device)
        start_setup = time.perf_counter()
        result = fn(indices, inputs, radius)
        synchronize(device)
        setup = time.perf_counter() - start_setup
        start = time.perf_counter()
        for _ in range(repetitions):
            result = fn(indices, inputs, radius)
        synchronize(device)
        elapsed = (time.perf_counter() - start)/repetitions
    else:
        synchronize(device)
        start_setup = time.perf_counter()
        indices = indices_cpu.to(device)
        inputs = inputs_cpu.to(device)
        result = fn(indices, inputs, radius)
        synchronize(device)
        result = result.cpu()
        setup = time.perf_counter() - start_setup
        start = time.perf_counter()
        for _ in range(repetitions):
            indices = indices_cpu.to(device)
            inputs = inputs_cpu.to(device)
            result = fn(indices, inputs, radius).cpu()
        synchronize(device)
        elapsed = (time.perf_counter() - start)/repetitions
    return result.detach().cpu().numpy(), setup, elapsed


def version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "unavailable"
    module = __import__(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--rhs", type=int, default=DEFAULT_RHS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/sparse_compact.csv"))
    args = parser.parse_args()
    if args.n < 1 or args.radius < 1 or args.rhs < 1 or args.repetitions < 1:
        raise SystemExit("n, radius, rhs, and repetitions must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    prepare_keops_runtime()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    dtype = torch.float64
    indices_cpu, inputs_cpu = make_inputs(args.n, args.rhs, dtype)
    expected = oracle(args.n, args.radius, inputs_cpu)
    backends = {"pytorch_dense": dense_torch, "keops": keops_torch}
    rows = []
    residencies = ("resident", "transfer") if args.device == "cuda" else ("resident",)
    for backend, function in backends.items():
        for residency in residencies:
            if args.device == "cuda":
                torch.cuda.empty_cache()
            try:
                result, setup, elapsed = timed(
                    function,
                    indices_cpu,
                    inputs_cpu,
                    device,
                    args.radius,
                    residency,
                    args.repetitions,
                )
                error = float(np.max(np.abs(result - expected)))
                relative_error = error/max(1.0, float(np.max(np.abs(expected))))
                if relative_error > 5.0e-7:
                    raise RuntimeError(f"independent oracle mismatch: {relative_error:g}")
                status = "pass"
            except (MemoryError, ImportError, OSError, RuntimeError) as exc:
                message = str(exc).replace("\n", " ")
                if "memory" in message.lower() or "out of memory" in message.lower():
                    status = "oom"
                else:
                    raise
                setup = math.nan
                elapsed = math.nan
                relative_error = math.nan
            rows.append(
                {
                    "backend": backend,
                    "device": args.device,
                    "residency": residency,
                    "n_samples": args.n,
                    "radius": args.radius,
                    "rhs": args.rhs,
                    "dtype": "float64",
                    "threads": args.threads,
                    "repetitions": args.repetitions,
                    "setup_seconds": setup,
                    "seconds_per_operation": elapsed,
                    "relative_error": relative_error,
                    "status": status,
                    "torch_version": torch.__version__,
                    "pykeops_version": version("pykeops"),
                    "cuda_device": (
                        torch.cuda.get_device_name(0)
                        if args.device == "cuda"
                        else "cpu"
                    ),
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
