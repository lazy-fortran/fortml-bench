#!/usr/bin/env python3
"""Correctness-gated XGBoost row/feature-sampling release lane.

The release fixture uses one depth-one squared-error tree.  This script
reimplements the local Park--Miller sampling stream, exhaustive split gains,
Newton leaf values, and full-data prediction in NumPy.  The app output is
accepted only after the independent oracle agrees; the CUDA row is a typed
refusal and is never inferred from the host timing.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


N, P = 12, 4
SUBSAMPLE, COLSAMPLE, SEED = 0.5, 0.5, 12345
LEARNING_RATE, L2 = 0.8, 1.0
FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "seconds", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = False
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        if name not in ignored_names:
            dirty = True
            break
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((N, P), dtype=np.float64)
    y = np.empty(N, dtype=np.float64)
    for i in range(1, N + 1):
        x[i - 1, 0] = i - 1
        x[i - 1, 1] = (3 * i + 1) % 11
        x[i - 1, 2] = (5 * i + 2) % 13
        x[i - 1, 3] = (7 * i + 3) % 17
        y[i - 1] = (
            1.4 * x[i - 1, 0] - 0.55 * x[i - 1, 1]
            + 0.2 * x[i - 1, 2] + 0.1 * x[i - 1, 3]
            + (0.7 if i % 2 == 0 else -0.4)
        )
    return x, y


def draw_without_replacement(n: int, fraction: float, state: int) -> tuple[np.ndarray, int]:
    modulus, multiplier = 2_147_483_647, 48_271
    permutation = list(range(n))
    count = max(1, min(n, int(np.ceil(fraction * n))))
    for i in range(count):
        state %= modulus
        if state <= 0:
            state = 1
        state = (multiplier * state) % modulus
        if state <= 0:
            state = 1
        j = i + state % (n - i)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    selected = np.sort(np.asarray(permutation[:count], dtype=np.int64))
    return selected, state


def oracle() -> tuple[np.ndarray, np.ndarray, float, int, int]:
    x, y = fixture()
    sample, state = draw_without_replacement(N, SUBSAMPLE, SEED)
    feature_sample, _ = draw_without_replacement(P, COLSAMPLE, state)
    feature_mask = np.zeros(P, dtype=bool)
    feature_mask[feature_sample] = True
    base = float(np.mean(y))
    gradient = base - y
    hessian = np.ones(N)
    total_g = float(np.sum(gradient[sample]))
    total_h = float(sample.size)
    best_gain, best_feature, best_threshold = 0.0, -1, 0.0
    best_left_weight = best_right_weight = 0.0
    for feature in np.flatnonzero(feature_mask):
        local = sample[np.argsort(x[sample, feature], kind="stable")]
        values = x[local, feature]
        left_g = left_h = 0.0
        for k in range(local.size - 1):
            left_g += gradient[local[k]]
            left_h += hessian[local[k]]
            if values[k] >= values[k + 1]:
                continue
            right_g, right_h = total_g - left_g, total_h - left_h
            gain = 0.5 * (
                left_g * left_g / (left_h + L2)
                + right_g * right_g / (right_h + L2)
                - total_g * total_g / (total_h + L2)
            )
            if gain > best_gain:
                best_gain = gain
                best_feature = int(feature)
                best_threshold = 0.5 * (values[k] + values[k + 1])
                best_left_weight = -left_g / (left_h + L2)
                best_right_weight = -right_g / (right_h + L2)
    correction = np.full(N, -total_g / (total_h + L2), dtype=np.float64)
    if best_feature >= 0:
        correction = np.where(
            x[:, best_feature] < best_threshold, best_left_weight, best_right_weight,
        )
    prediction = base + LEARNING_RATE * correction
    importance = np.zeros(P, dtype=np.float64)
    if best_feature >= 0:
        importance[best_feature] = 1.0
    return prediction, importance, base, best_feature, int(3 if best_feature >= 0 else 1)


def build_probe(fortml: Path, fixture_path: Path) -> tuple[str, float]:
    build = subprocess.run(
        ["fo", "build", "--flag", "-O2"], cwd=fortml,
        capture_output=True, text=True, check=False,
    )
    if build.returncode:
        raise RuntimeError(build.stderr.strip() or build.stdout.strip())
    archives = list((fortml / "build" / "fo" / "lib").glob("*.a"))
    if not archives:
        raise RuntimeError("fo build produced no archive")
    archive = max(archives, key=lambda path: path.stat().st_mtime_ns)
    module_dir = fortml / "build" / "fo" / "mod"
    compiler = shlex.split(os.environ.get("FO_FC", "gfortran"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError(f"Fortran compiler unavailable: {compiler!r}")
    with tempfile.TemporaryDirectory(prefix="fortml-xgb-sampling-", dir="/mnt/storage") as directory:
        directory_path = Path(directory)
        source = directory_path / fixture_path.name
        executable = directory_path / "xgboost_sampling_probe"
        source.write_bytes(fixture_path.read_bytes())
        command = compiler + ["-O2", "-ffree-line-length-none", "-I", str(module_dir),
                              str(source), str(archive), "-o", str(executable)]
        link = subprocess.run(command, cwd=fortml, capture_output=True, text=True, check=False)
        if link.returncode:
            raise RuntimeError(link.stderr.strip() or link.stdout.strip())
        started = time.perf_counter()
        run = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        if run.returncode:
            raise RuntimeError(run.stderr.strip() or run.stdout.strip())
        return run.stdout, elapsed


def parse(stdout: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0].startswith("xgb_sampling_"):
            rows[fields[0]] = fields[1:]
    required = {"xgb_sampling_base", "xgb_sampling_node_count", "xgb_sampling_depth",
                "xgb_sampling_importance", "xgb_sampling_prediction", "xgb_sampling_cuda"}
    if set(rows) != required:
        raise RuntimeError(f"unexpected probe rows: {sorted(rows)}\n{stdout}")
    return rows


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/xgboost_sampling.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    fixture_path = root / "fixtures" / "xgboost_sampling_probe.f90"
    expected_prediction, expected_importance, expected_base, best_feature, expected_nodes = oracle()
    stdout, elapsed = build_probe(fortml, fixture_path)
    observed = parse(stdout)
    base = float(observed["xgb_sampling_base"][0].replace("D", "E"))
    nodes, depth = int(observed["xgb_sampling_node_count"][0]), int(observed["xgb_sampling_depth"][0])
    importance = np.asarray([float(value.replace("D", "E")) for value in observed["xgb_sampling_importance"]])
    prediction = np.asarray([float(value.replace("D", "E")) for value in observed["xgb_sampling_prediction"]])
    if importance.size != P or prediction.size != N:
        raise RuntimeError("probe shape mismatch")
    if not np.isclose(base, expected_base, rtol=0.0, atol=2e-13):
        raise RuntimeError(f"base mismatch: app={base}, oracle={expected_base}")
    error = float(max(np.max(np.abs(prediction - expected_prediction)),
                      np.max(np.abs(importance - expected_importance))))
    if nodes != expected_nodes or depth != (1 if expected_nodes == 3 else 0) or error > 3e-12:
        raise RuntimeError(f"sampling oracle mismatch nodes={nodes}/{expected_nodes}, depth={depth}, error={error:.3e}")
    cuda_code = int(observed["xgb_sampling_cuda"][0])
    if cuda_code != 3:
        raise RuntimeError(f"expected FORTNUM_NOT_IMPLEMENTED=3, got {cuda_code}")
    source_revision = revision(fortml)
    output = args.output.resolve()
    details = {
        "oracle": "independent NumPy Park-Miller sampling, Newton stump, and prediction",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": source_revision, "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O2",
    }
    records = [
        row(details, workload="xgboost_sampling", phase="fit_predict", backend="fortml",
            device="cpu", status="pass", metric="max_abs_error", value=error,
            max_abs_error=error, seconds=elapsed,
            notes=f"subsample={SUBSAMPLE}; colsample_bytree={COLSAMPLE}; seed={SEED}; selected_feature={best_feature + 1}"),
        row(details, workload="xgboost_sampling", phase="device_capability", backend="fortml",
            device="cuda", status="unavailable", metric="predict", value="nan",
            max_abs_error="nan", oracle="typed device contract", seconds="",
            notes="FORTNUM_NOT_IMPLEMENTED; tree growth/prediction has no resident CUDA kernel"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
