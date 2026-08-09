#!/usr/bin/env python3
"""Independent MLP-regressor training and FortOpt L-BFGS-B benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_outputs", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
TOLERANCE = 2.0e-10


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[-2.0], [-1.5], [-1.0], [-0.25], [0.25], [1.0], [1.5], [2.0]], dtype=np.float64)
    return x, 0.8*x - 0.35


def initialize(seed: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale1 = np.sqrt(6.0/4.0)
    scale2 = np.sqrt(6.0/4.0)
    w1 = np.empty((1, 3), dtype=np.float64)
    for j in range(1, 4):
        index = 1 + 1*(j - 1)
        w1[0, j - 1] = scale1*np.sin(seed + 1009*1 + 9176*index)
    b1 = np.array([0.01*scale1*np.sin(seed + 1009*1 + 7919*j) for j in range(1, 4)])
    w2 = np.empty((3, 1), dtype=np.float64)
    for j in range(1, 2):
        for i in range(1, 4):
            index = i + 3*(j - 1)
            w2[i - 1, j - 1] = scale2*np.sin(seed + 1009*2 + 9176*index)
    b2 = np.array([0.01*scale2*np.sin(seed + 1009*2 + 7919)])
    return w1, b1, w2, b2


def numpy_adam_training(x: np.ndarray, y: np.ndarray) -> float:
    w1, b1, w2, b2 = initialize()
    m = [np.zeros_like(v) for v in (w1, b1, w2, b2)]
    v = [np.zeros_like(v) for v in (w1, b1, w2, b2)]
    for step in range(1, 25):
        z1 = x @ w1 + b1
        h1 = np.tanh(z1)
        pred = h1 @ w2 + b2
        residual = pred - y
        d2 = residual / x.shape[0]
        gw2 = h1.T @ d2
        gb2 = d2.sum(axis=0)
        dh = d2 @ w2.T
        dz = dh*(1.0 - h1*h1)
        gw1 = x.T @ dz
        gb1 = dz.sum(axis=0)
        for i, (param, grad) in enumerate(zip((w1, b1, w2, b2), (gw1, gb1, gw2, gb2))):
            m[i] = 0.9*m[i] + 0.1*grad
            v[i] = 0.999*v[i] + 0.001*grad*grad
            param -= 0.02*(m[i]/(1.0 - 0.9**step))/(
                np.sqrt(v[i]/(1.0 - 0.999**step)) + 1.0e-8)
    pred = np.tanh(x @ w1 + b1) @ w2 + b2
    return float(np.mean((pred - y)**2))


def linear_oracle(x: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack((x[:, 0], np.ones(x.shape[0])))
    beta, *_ = np.linalg.lstsq(design, y[:, 0], rcond=None)
    return float(np.mean((design @ beta - y[:, 0])**2))


def run_release_app(fortml: Path) -> dict[str, tuple[float, float]]:
    env = os.environ.copy()
    env.update({"FO_FC": env.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=env,
                   check=True, capture_output=True, text=True)
    started = time.perf_counter()
    completed = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_mlp_regressor"],
                               cwd=fortml, env=env, check=True, capture_output=True, text=True)
    wall = time.perf_counter() - started
    pattern = re.compile(r"^(mlp_regressor_(?:train|lbfgsb)),\s*(\d+),\s*(\d+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$")
    rows: dict[str, tuple[float, float]] = {}
    for line in completed.stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            rows[match.group(1)] = (float(match.group(4)), float(match.group(5)))
    if set(rows) != {"mlp_regressor_train", "mlp_regressor_lbfgsb"}:
        raise RuntimeError(f"release app omitted rows: {sorted(rows)}; output={completed.stdout!r}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_regressor.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/MLP_REGRESSOR.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    x, y = fixture()
    expected_train = numpy_adam_training(x, y)
    expected_lbfgs = linear_oracle(x, y)
    measured = run_release_app(fortml)
    train_seconds, train_value = measured["mlp_regressor_train"]
    lbfgs_seconds, lbfgs_value = measured["mlp_regressor_lbfgsb"]
    train_error = abs(train_value - expected_train)
    lbfgs_error = abs(lbfgs_value - expected_lbfgs)
    if train_error > TOLERANCE or lbfgs_error > TOLERANCE:
        raise RuntimeError(f"MLP regressor oracle mismatch: train={train_error:g}, lbfgsb={lbfgs_error:g}")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(), args.report.resolve())),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows = [
        {**details, "workload": "mlp_regressor", "phase": "adam_training", "backend": "fortml",
         "device": "cpu", "status": "pass", "n_samples": x.shape[0], "n_features": 1,
         "n_outputs": 1, "seconds_per_operation": train_seconds, "metric": "mse",
         "value": train_value, "max_abs_error": train_error,
         "oracle": "independent NumPy tanh-MLP/Adam recurrence",
         "notes": "topology=[1,3,1]; 24 full-batch epochs; seed=17"},
        {**details, "workload": "mlp_regressor", "phase": "lbfgsb_training", "backend": "fortml",
         "device": "cpu", "status": "pass", "n_samples": x.shape[0], "n_features": 1,
         "n_outputs": 1, "seconds_per_operation": lbfgs_seconds, "metric": "mse",
         "value": lbfgs_value, "max_abs_error": lbfgs_error,
         "oracle": "independent NumPy affine least-squares optimum",
         "notes": "topology=[1,1]; FortOpt L-BFGS-B; exact objective derivative"},
        {**details, "workload": "mlp_regressor", "phase": "device_capability", "backend": "fortml",
         "device": "cuda", "status": "unavailable", "n_samples": x.shape[0], "n_features": 1,
         "n_outputs": 1, "seconds_per_operation": "", "metric": "status", "value": 3.0,
         "max_abs_error": "", "oracle": "typed_device_contract",
         "notes": "FORTNUM_NOT_IMPLEMENTED; no host fallback"},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# MLP regressor\n\n"
        "The release application is checked against an independent NumPy tanh/Adam recurrence and an affine least-squares oracle for the exact FortOpt L-BFGS-B path. The CUDA row is an explicit typed refusal until resident trainer state is implemented.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Adam training MSE error: `{train_error:.6g}`\n"
        f"- L-BFGS-B MSE error: `{lbfgs_error:.6g}`\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n"
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
