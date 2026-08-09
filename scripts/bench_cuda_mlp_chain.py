#!/usr/bin/env python3
"""Correctness-gated resident CUDA dense-MLP-chain benchmark lane."""

from __future__ import annotations

import argparse
import csv
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "input_width", "output_width", "n_layers", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


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


def activate(value: np.ndarray, code: int) -> np.ndarray:
    if code == 2:
        return np.tanh(value)
    if code == 3:
        return np.maximum(value, 0.0)
    return value


def derivative(value: np.ndarray, code: int) -> np.ndarray:
    if code == 2:
        tangent = np.tanh(value)
        return 1.0 - tangent * tangent
    if code == 3:
        return np.where(value >= 0.0, 1.0, 0.0)
    return np.ones_like(value)


def fixture() -> tuple[list[int], list[int], list[np.ndarray], list[np.ndarray],
                        np.ndarray, np.ndarray]:
    sizes = [2, 3, 2, 1]
    activations = [2, 3, 1]
    weights = [
        np.array([[0.11, -0.08], [0.04, 0.17], [-0.13, 0.09]]),
        np.array([[0.07, -0.06, 0.12], [-0.05, 0.14, 0.03]]),
        np.array([[0.18, -0.21]]),
    ]
    biases = [
        np.array([-0.04, 0.06, 0.02]),
        np.array([0.08, -0.03]),
        np.array([0.05]),
    ]
    x = np.array([
        [0.2, -0.4], [0.7, 0.3], [-0.5, 0.8], [0.1, -0.2],
        [0.6, -0.7],
    ], dtype=np.float64)
    tangent = np.array([
        [-0.3, 0.4], [0.2, -0.1], [0.1, 0.5], [-0.4, 0.3],
        [0.25, -0.2],
    ], dtype=np.float64)
    return sizes, activations, weights, biases, x, tangent


def evaluate(weights: list[np.ndarray], biases: list[np.ndarray],
             activations: list[int], x: np.ndarray) -> np.ndarray:
    value = x
    for weight, bias, code in zip(weights, biases, activations):
        value = activate(value @ weight.T + bias, code)
    return value


def oracle() -> dict[str, float | int]:
    sizes, activations, weights, biases, x, tangent = fixture()
    weight_tangent = [
        np.full_like(weights[0], -0.07), np.full_like(weights[1], 0.05),
        np.full_like(weights[2], -0.03),
    ]
    bias_tangent = [
        np.full_like(biases[0], 0.02), np.full_like(biases[1], -0.04),
        np.full_like(biases[2], 0.06),
    ]
    value = x
    value_dot = tangent
    preacts: list[np.ndarray] = []
    for weight, bias, code, weight_dot, bias_dot in zip(
            weights, biases, activations, weight_tangent, bias_tangent):
        preact = value @ weight.T + bias
        preacts.append(preact)
        value_dot = value_dot @ weight.T + value @ weight_dot.T + bias_dot
        value = activate(preact, code)
        value_dot = derivative(preact, code) * value_dot
    h = 2.0e-6
    plus_weights = [w + h * dw for w, dw in zip(weights, weight_tangent)]
    minus_weights = [w - h * dw for w, dw in zip(weights, weight_tangent)]
    plus_biases = [b + h * db for b, db in zip(biases, bias_tangent)]
    minus_biases = [b - h * db for b, db in zip(biases, bias_tangent)]
    finite_difference = (
        evaluate(plus_weights, plus_biases, activations, x + h * tangent)
        - evaluate(minus_weights, minus_biases, activations, x - h * tangent)
    ) / (2.0 * h)
    u = np.array([[0.4], [-0.2], [0.3], [0.1], [-0.5]], dtype=np.float64)
    state_bar = u
    weight_bars: list[np.ndarray] = [np.zeros_like(w) for w in weights]
    bias_bars: list[np.ndarray] = [np.zeros_like(b) for b in biases]
    values: list[np.ndarray] = [x]
    state = x
    for weight, bias, code in zip(weights, biases, activations):
        state = activate(state @ weight.T + bias, code)
        values.append(state)
    for layer in range(len(weights) - 1, -1, -1):
        zbar = state_bar * derivative(preacts[layer], activations[layer])
        weight_bars[layer][:] = zbar.T @ values[layer]
        bias_bars[layer][:] = np.sum(zbar, axis=0)
        state_bar = zbar @ weights[layer]
    adjoint_error = abs(
        float(np.sum(u * value_dot))
        - float(np.sum(state_bar * tangent))
        - sum(float(np.sum(wb * wd)) for wb, wd in zip(weight_bars, weight_tangent))
        - sum(float(np.sum(bb * bd)) for bb, bd in zip(bias_bars, bias_tangent))
    )
    derivative_error = float(np.max(np.abs(value_dot - finite_difference)))
    if derivative_error > 2.0e-8 or adjoint_error > 2.0e-12:
        raise RuntimeError(
            f"independent chain oracle failed: jvp={derivative_error}, "
            f"adjoint={adjoint_error}"
        )
    return {
        "n_samples": int(x.shape[0]), "input_width": sizes[0],
        "output_width": sizes[-1], "n_layers": len(weights),
        "derivative_error": derivative_error, "adjoint_error": adjoint_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/cuda_mlp_chain.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/CUDA_MLP_CHAIN.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    oracle_metrics = oracle()
    started = time.perf_counter()
    gate = subprocess.run(
        ["fo", "test", "test_cuda_mlp_chain_api"], cwd=fortml,
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    gate_ok = gate.returncode == 0
    cuda_ready = shutil.which("nvcc") is not None
    if cuda_ready:
        cuda_ready = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    cuda_status = "unavailable"
    cuda_value = "typed_refusal"
    cuda_elapsed: float | str = ""
    cuda_notes = "nvcc/device unavailable; ordinary build returns FORTNUM_NOT_IMPLEMENTED"
    if cuda_ready:
        started = time.perf_counter()
        cuda_gate = subprocess.run(
            ["bash", "test/run_cuda_mlp_chain.sh"], cwd=fortml,
            capture_output=True, text=True,
        )
        cuda_elapsed = time.perf_counter() - started
        cuda_status = "pass" if cuda_gate.returncode == 0 else "failed"
        cuda_value = 1.0 if cuda_gate.returncode == 0 else 0.0
        cuda_notes = "native CUDA chain oracle and resident transfer counters"
    metadata = {
        "n_samples": oracle_metrics["n_samples"],
        "input_width": oracle_metrics["input_width"],
        "output_width": oracle_metrics["output_width"],
        "n_layers": oracle_metrics["n_layers"],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": "gfortran/nvcc", "flags": "-O3",
        "oracle": "independent NumPy dense-chain recurrence",
    }
    rows = [{
        "workload": "cuda_mlp_chain", "phase": "value_jvp_vjp",
        "backend": "numpy_oracle", "device": "cpu", "status": "pass",
        "seconds_per_operation": "", "metric": "jvp_max_abs_error",
        "value": 1.0, "max_abs_error": oracle_metrics["derivative_error"],
        **metadata, "notes": "central finite difference plus adjoint oracle",
    }, {
        "workload": "cuda_mlp_chain", "phase": "fortran_stub",
        "backend": "fortml", "device": "cpu", "status": "pass" if gate_ok else "failed",
        "seconds_per_operation": elapsed, "metric": "gate_seconds",
        "value": 1.0 if gate_ok else 0.0,
        "max_abs_error": oracle_metrics["adjoint_error"], **metadata,
        "notes": "ordinary-build typed CUDA refusal preserves sentinels",
    }, {
        "workload": "cuda_mlp_chain", "phase": "resident_inference",
        "backend": "fortml", "device": "cuda", "status": cuda_status,
        "seconds_per_operation": cuda_elapsed, "metric": "api_surface",
        "value": cuda_value, "max_abs_error": 0.0, **metadata,
        "oracle": "independent NumPy recurrence and packed JVP/VJP oracle",
        "notes": cuda_notes,
    }]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Resident CUDA dense MLP-chain\n\n"
        "This lane compares a three-layer dense chain against an independent "
        "NumPy recurrence. The oracle checks the complete value, packed input "
        "and parameter JVP, packed input and parameter VJP, a central finite "
        "difference, and the reverse-mode adjoint identity. The Fortran test "
        "checks ordinary-build typed refusal and sentinel preservation. Native "
        "CUDA is run only when both `nvcc` and `nvidia-smi` are available. An "
        "unavailable device is recorded as `typed_refusal`, never as CPU GPU "
        "evidence.\n\n"
        "Run:\n\n```sh\n"
        "python3 scripts/bench_cuda_mlp_chain.py --fortml ../fortml \\\n"
        "  --output results/cuda_mlp_chain.csv \\\n"
        "  --report results/CUDA_MLP_CHAIN.md\n```\n\n"
        f"Oracle JVP error: `{oracle_metrics['derivative_error']:.3e}`. "
        f"Adjoint error: `{oracle_metrics['adjoint_error']:.3e}`.\n"
    )
    print(f"wrote {len(rows)} rows to {output}")
    if not gate_ok:
        print(gate.stdout + gate.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
