from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.bench_rbf_mvm import (
        dense_torch,
        gpytorch_keops,
        keops_torch,
        make_inputs,
        oracle,
        prepare_keops_runtime,
        synchronize,
    )
except ModuleNotFoundError:
    from bench_rbf_mvm import (
        dense_torch,
        gpytorch_keops,
        keops_torch,
        make_inputs,
        oracle,
        prepare_keops_runtime,
        synchronize,
    )


BACKENDS = {
    "pytorch_dense": dense_torch,
    "keops": keops_torch,
    "gpytorch_keops": gpytorch_keops,
}


def event_value(event, name: str) -> float:
    value = getattr(event, name, 0.0)
    return float(value) if value is not None else 0.0


def profile_backend(
    backend: str,
    function,
    points: torch.Tensor,
    vector: torch.Tensor,
    expected: np.ndarray,
    device: torch.device,
) -> list[dict[str, object]]:
    for _ in range(2):
        result = function(points, vector)
        synchronize(device)
    actual = result.detach().cpu().numpy()
    scale = max(1.0, float(np.max(np.abs(expected))))
    relative_error = float(np.max(np.abs(actual - expected))) / scale
    tolerance = 5.0e-5 if points.dtype == torch.float32 else 1.0e-6
    if relative_error > tolerance:
        raise RuntimeError(
            f"{backend} independent oracle mismatch: {relative_error:g}"
        )

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        result = function(points, vector)
        synchronize(device)
        profiler.step()
    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        rows.append(
            {
                "backend": backend,
                "device": device.type,
                "n_samples": points.shape[0],
                "n_features": points.shape[1],
                "dtype": str(points.dtype).removeprefix("torch."),
                "relative_error": relative_error,
                "operation": event.key,
                "calls": event.count,
                "self_cpu_us": event_value(event, "self_cpu_time_total"),
                "total_cpu_us": event_value(event, "cpu_time_total"),
                "self_cuda_us": event_value(event, "self_device_time_total"),
                "total_cuda_us": event_value(event, "device_time_total"),
                "self_cpu_bytes": event.self_cpu_memory_usage,
                "self_cuda_bytes": event.self_device_memory_usage,
                "input_shapes": event.input_shapes,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/operation_profile_cpu.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.n, args.d, args.threads) < 1:
        raise SystemExit("n, d, and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    prepare_keops_runtime()
    torch.set_num_threads(args.threads)
    dtype = torch.float64
    device = torch.device(args.device)
    points_cpu, vector_cpu = make_inputs(args.n, args.d, dtype, torch.device("cpu"))
    expected = oracle(points_cpu, vector_cpu)
    points = points_cpu.to(device)
    vector = vector_cpu.to(device)
    rows = []
    for backend, function in BACKENDS.items():
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            rows.extend(
                profile_backend(
                    backend, function, points, vector, expected, device
                )
            )
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    fields = [
        "backend",
        "device",
        "n_samples",
        "n_features",
        "dtype",
        "relative_error",
        "operation",
        "calls",
        "self_cpu_us",
        "total_cpu_us",
        "self_cuda_us",
        "total_cuda_us",
        "self_cpu_bytes",
        "self_cuda_bytes",
        "input_shapes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for backend in BACKENDS:
        selected = [row for row in rows if row["backend"] == backend]
        selected.sort(
            key=lambda row: max(
                float(row["self_cuda_us"]), float(row["self_cpu_us"])
            ),
            reverse=True,
        )
        print(f"{backend}:")
        for row in selected[:12]:
            cpu = float(row["self_cpu_us"])
            cuda = float(row["self_cuda_us"])
            if math.isclose(cpu, 0.0) and math.isclose(cuda, 0.0):
                continue
            print(
                f"  {row['operation']} calls={row['calls']} "
                f"self_cpu_us={cpu:.1f} self_cuda_us={cuda:.1f}"
            )
    print(args.output)


if __name__ == "__main__":
    main()
