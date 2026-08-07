#!/usr/bin/env python3
"""Correctness-gated benchmark for FortML's composed MLP module tree.

NumPy forms the two dense stages and all first/second products independently.
The Fortran release app must match those values before its timings are kept.
CUDA is recorded as an explicit unavailable capability until a resident fused
chain kernel exists.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "dimensions",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N_SAMPLES = 64
EPS = 1.0e-6


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, ...]:
    x = np.array([[math.sin(0.17 * i), math.cos(0.11 * i)]
                  for i in range(1, N_SAMPLES + 1)], dtype=np.float64)
    dx = np.array([[0.03 * math.cos(0.17 * i), -0.02 * math.sin(0.11 * i)]
                   for i in range(1, N_SAMPLES + 1)], dtype=np.float64)
    u = (0.2 + 0.01 * (np.arange(1, N_SAMPLES + 1) % 7))[:, None]
    theta = np.array((0.30, -0.20, 0.10, 0.40, -0.50, 0.60, 0.20, -0.10,
                      0.05, -0.03, 0.07, -0.09, 0.11, -0.13, 0.17,
                      -0.19, 0.23), dtype=np.float64)
    dtheta = 0.01 * np.sin(np.arange(1, 18, dtype=np.float64))
    return x, dx, u, theta, dtheta


def forward(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    w1 = theta[:8].reshape((2, 4), order="F")
    b1 = theta[8:12]
    w2 = theta[12:16].reshape((4, 1), order="F")
    b2 = theta[16:17]
    return (x @ w1 + b1) @ w2 + b2


def vjp(theta: np.ndarray, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w1 = theta[:8].reshape((2, 4), order="F")
    b1 = theta[8:12]
    w2 = theta[12:16].reshape((4, 1), order="F")
    z1 = x @ w1 + b1
    dz1 = u @ w2.T
    packed = np.concatenate((
        (x.T @ dz1).reshape(-1, order="F"), dz1.sum(axis=0),
        (z1.T @ u).reshape(-1, order="F"), u.sum(axis=0),
    ))
    return packed, dz1 @ w1.T


def products() -> dict[tuple[str, int], float]:
    x, dx, u, theta, dtheta = fixture()
    w1 = theta[:8].reshape((2, 4), order="F")
    b1 = theta[8:12]
    w2 = theta[12:16].reshape((4, 1), order="F")
    db1 = dtheta[8:12]
    dw1 = dtheta[:8].reshape((2, 4), order="F")
    dw2 = dtheta[12:16].reshape((4, 1), order="F")
    db2 = dtheta[16:17]
    z1 = x @ w1 + b1
    dz1 = dx @ w1 + x @ dw1 + db1
    dy = dz1 @ w2 + z1 @ dw2 + db2
    theta_bar, x_bar = vjp(theta, x, u)
    plus_bar, plus_x = vjp(theta + EPS * dtheta, x + EPS * dx, u)
    minus_bar, minus_x = vjp(theta - EPS * dtheta, x - EPS * dx, u)
    theta_hvp = (plus_bar - minus_bar) / (2.0 * EPS)
    x_hvp = (plus_x - minus_x) / (2.0 * EPS)
    expected: dict[tuple[str, int], float] = {}
    y = forward(theta, x)
    for i in range(N_SAMPLES):
        expected["prediction", i + 1] = float(y[i, 0])
        expected["jvp", i + 1] = float(dy[i, 0])
        expected["x_vjp", i + 1] = float(x_bar[i, 0])
        expected["x_hvp", i + 1] = float(x_hvp[i, 0])
    for i, value in enumerate(theta_bar, 1):
        expected["parameter_vjp", i] = float(value)
    for i, value in enumerate(theta_hvp, 1):
        expected["parameter_hvp", i] = float(value)
    return expected


def parse_oracle(path: Path) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], float] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values[row["quantity"], int(row["index"])] = float(row["value"])
    return values


def row(details: dict[str, str], **updates: object) -> dict[str, str]:
    output = {field: "" for field in FIELDS}
    output.update(details)
    output.update({key: str(value) for key, value in updates.items()})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_chain.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_chain")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected = products()
    with tempfile.TemporaryDirectory(prefix="fortml-chain-", dir=fortml / "build") as directory:
        oracle_path = Path(directory) / "oracle.csv"
        environment = os.environ.copy()
        environment["FORTML_BENCH_MLP_CHAIN_ORACLE"] = str(oracle_path)
        environment["FORTML_BENCH_ORACLE_ONLY"] = "1"
        subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                       env=environment, check=True, capture_output=True, text=True)
        actual = parse_oracle(oracle_path)
    if set(actual) != set(expected):
        raise RuntimeError("FortML chain oracle keys differ from NumPy fixture")
    max_error = max(abs(actual[key] - value) for key, value in expected.items())
    if max_error > 3.0e-8:
        raise RuntimeError(f"FortML chain product oracle mismatch: {max_error:.3e}")

    completed = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                               check=True, capture_output=True, text=True)
    timings: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 2 and fields[0].startswith("mlp_chain_"):
            timings[fields[0].removeprefix("mlp_chain_")] = float(fields[1])
    if set(timings) != {"predict", "jvp", "vjp", "hvp"}:
        raise RuntimeError(f"missing chain timings: {timings}")

    details = {
        "workload": "mlp_chain", "backend": "fortml", "device": "cpu",
        "status": "pass", "dimensions": "2->4->1", "repetitions": "2048",
        "oracle": "independent NumPy two-stage chain and finite-difference HVP",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
        "max_abs_error": f"{max_error:.17e}", "notes": "named encoder/head parameter tree",
    }
    rows = []
    for phase, seconds in timings.items():
        rows.append(row(details, phase=phase, seconds_per_operation=f"{seconds:.17e}",
                        metric="seconds_per_operation", value=f"{seconds:.17e}"))
    rows.append(row(details, phase="device_capability", device="cuda", status="unavailable",
                    seconds_per_operation="", metric="", value="", max_abs_error="",
                    oracle="typed_device_contract",
                    notes="mlp_chain device_supported(CUDA)=false; no resident fused chain"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
