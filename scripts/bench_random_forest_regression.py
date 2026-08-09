#!/usr/bin/env python3
"""Correctness-gated weighted random-forest regression benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


N_SAMPLES, N_FEATURES, N_OUTPUTS, N_QUERY, N_TREES = 8, 2, 2, 4, 25
MAX_DEPTH, MIN_LEAF, SEED = 3, 1, 5489
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_outputs", "n_query", "n_trees", "max_depth",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        if line[3:].strip() not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.column_stack((
        [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        np.full(N_SAMPLES, 0.25),
    ))
    targets = np.column_stack((
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        0.5 + 2.0*np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ))
    query = np.column_stack(([-2.7, -0.123, 0.123, 2.7], np.full(N_QUERY, 0.25)))
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0])
    return x, targets, query, weights


def uniform(state: int) -> tuple[int, float]:
    state = (48271*state) % 2147483647
    return state, state/2147483647.0


def bootstrap(n: int, state: int) -> tuple[int, np.ndarray]:
    indices = np.empty(n, dtype=np.int64)
    for row in range(n):
        state, value = uniform(state)
        indices[row] = min(n - 1, int(n*value))
    return state, indices


@dataclass
class Node:
    feature: int = -1
    threshold: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None
    value: float = 0.0


def weighted_mean(y: np.ndarray, w: np.ndarray, rows: np.ndarray) -> float:
    return float(np.sum(w[rows]*y[rows])/np.sum(w[rows]))


def weighted_sse(y: np.ndarray, w: np.ndarray, rows: np.ndarray) -> float:
    mean = weighted_mean(y, w, rows)
    return float(np.sum(w[rows]*(y[rows]-mean)**2))


def best_split(x: np.ndarray, y: np.ndarray, w: np.ndarray, rows: np.ndarray,
               min_leaf: int) -> tuple[int, float, float] | None:
    best: tuple[int, float, float] | None = None
    for feature in range(x.shape[1]):
        # Stable insertion sort matches the Fortran subset ordering.
        order = sorted(range(len(rows)), key=lambda position: x[rows[position], feature])
        for k in range(1, len(rows)):
            if k < min_leaf or len(rows)-k < min_leaf:
                continue
            left = rows[np.array(order[:k], dtype=np.int64)]
            right = rows[np.array(order[k:], dtype=np.int64)]
            if x[left[-1], feature] >= x[right[0], feature]:
                continue
            threshold = 0.5*(x[left[-1], feature] + x[right[0], feature])
            sse = weighted_sse(y, w, left) + weighted_sse(y, w, right)
            if best is None or sse < best[2]:
                best = (feature, threshold, sse)
    return best


def fit_tree(x: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int,
             min_leaf: int, rows: np.ndarray, level: int = 0) -> Node:
    node = Node(value=weighted_mean(y, w, rows))
    if level >= depth or len(rows) < 2*min_leaf:
        return node
    parent = weighted_sse(y, w, rows)
    split = best_split(x, y, w, rows, min_leaf)
    if split is None or split[2] >= parent:
        return node
    feature, threshold, _ = split
    left = rows[x[rows, feature] < threshold]
    right = rows[x[rows, feature] >= threshold]
    if len(left) < min_leaf or len(right) < min_leaf:
        return node
    node.feature = feature
    node.threshold = threshold
    node.left = fit_tree(x, y, w, depth, min_leaf, left, level+1)
    node.right = fit_tree(x, y, w, depth, min_leaf, right, level+1)
    return node


def predict_tree(node: Node, x: np.ndarray) -> np.ndarray:
    result = np.empty(x.shape[0])
    for i, row in enumerate(x):
        current = node
        while current.feature >= 0:
            current = current.left if row[current.feature] < current.threshold else current.right
        result[i] = current.value
    return result


def split_count(node: Node, counts: np.ndarray) -> None:
    if node.feature < 0:
        return
    counts[node.feature] += 1.0
    split_count(node.left, counts)  # type: ignore[arg-type]
    split_count(node.right, counts)  # type: ignore[arg-type]


def oracle(x: np.ndarray, targets: np.ndarray, query: np.ndarray,
           weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = SEED
    predictions = np.zeros((N_QUERY, N_OUTPUTS))
    importances = np.zeros(N_FEATURES)
    trees: list[list[Node]] = [[] for _ in range(N_OUTPUTS)]
    for _ in range(N_TREES):
        state, indices = bootstrap(N_SAMPLES, state)
        x_boot = x[indices]
        w_boot = weights[indices]
        for output in range(N_OUTPUTS):
            tree = fit_tree(
                x_boot, targets[indices, output], w_boot, MAX_DEPTH, MIN_LEAF,
                np.arange(N_SAMPLES, dtype=np.int64),
            )
            trees[output].append(tree)
            predictions[:, output] += predict_tree(tree, query)
            split_count(tree, importances)
    predictions /= N_TREES
    if np.sum(importances) > 0:
        importances /= np.sum(importances)
    return predictions, importances, state


def parse_app(stdout: str) -> tuple[np.ndarray, dict[str, float], str]:
    predictions = np.full((N_QUERY, N_OUTPUTS), np.nan)
    metrics: dict[str, float] = {}
    cuda = ""
    for line in stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if not fields or not fields[0]:
            continue
        if fields[0] == "random_forest_regression_cuda":
            cuda = fields[1]
        elif fields[0] == "random_forest_regression_prediction":
            predictions[int(fields[1])-1, int(fields[2])-1] = float(fields[3])
        elif fields[0].startswith("random_forest_regression_"):
            metrics[fields[0]] = float(fields[1])
    if np.isnan(predictions).any():
        raise RuntimeError("release app omitted a random-forest prediction")
    return predictions, metrics, cuda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/random_forest_regression.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/RANDOM_FOREST_REGRESSION.md"))
    parser.add_argument("--no-build", action="store_true",
                        help="reuse a previously built release app")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    x, targets, query, weights = fixture()
    expected, expected_importances, _ = oracle(x, targets, query, weights)
    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    if not args.no_build:
        subprocess.run(
            ["fo", "build", "--flag", "-O3"], cwd=fortml, env=env, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_random_forest_regression"],
        cwd=fortml, env=env, check=True, capture_output=True, text=True,
    )
    wall_seconds = time.perf_counter() - started
    actual, metrics, cuda = parse_app(completed.stdout)
    prediction_error = float(np.max(np.abs(actual - expected)))
    if prediction_error > 3.0e-12:
        raise RuntimeError(f"random-forest NumPy prediction mismatch: {prediction_error:.3e}")
    if abs(metrics["random_forest_regression_stage_error"]) > 3.0e-14:
        raise RuntimeError("staged prefix is inconsistent with final prediction")
    if abs(metrics["random_forest_regression_importance_sum"] - 1.0) > 3.0e-14:
        raise RuntimeError("feature importance does not normalize")
    if metrics["random_forest_regression_jvp_max"] != 0.0:
        raise RuntimeError("piecewise-constant JVP oracle failed")
    if cuda != "unavailable":
        raise RuntimeError(f"unexpected CUDA contract: {cuda!r}")

    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(), args.report.resolve())),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []
    def row(**values: object) -> dict[str, object]:
        result: dict[str, object] = {field: "" for field in FIELDS}
        result.update({
            "workload": "random_forest_regression", "backend": "fortml",
            "device": "cpu", "status": "pass", "n_samples": N_SAMPLES,
            "n_features": N_FEATURES, "n_outputs": N_OUTPUTS, "n_query": N_QUERY,
            "n_trees": N_TREES, "max_depth": MAX_DEPTH,
            "oracle": "independent NumPy weighted CART bootstrap replay", **details,
        })
        result.update(values)
        return result

    rows.append(row(phase="predict", seconds_per_operation=metrics[
        "random_forest_regression_predict_seconds"], metric="max_abs_error",
        value=float(np.linalg.norm(actual)), max_abs_error=prediction_error,
        notes="complete scalar and multi-output prediction oracle"))
    rows.append(row(phase="fit", seconds_per_operation=metrics[
        "random_forest_regression_fit_seconds"], metric="stage_error",
        value=metrics["random_forest_regression_stage_error"], max_abs_error=0.0,
        notes="staged prefix final state equals predict"))
    rows.append(row(phase="feature_importances", seconds_per_operation="",
                    metric="sum", value=metrics["random_forest_regression_importance_sum"],
                    max_abs_error=0.0, notes="normalized split-frequency diagnostic"))
    rows.append(row(phase="predict_jvp", seconds_per_operation="",
                    metric="max_abs", value=metrics["random_forest_regression_jvp_max"],
                    max_abs_error=0.0, notes="zero away from split surfaces"))
    rows.append(row(phase="predict", device="cuda", status="unavailable",
                    metric="status", value=3.0, oracle="typed_device_contract",
                    notes="FORTNUM_NOT_IMPLEMENTED; no host fallback"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Random-forest regression\n\n"
        "This correctness-gated lane replays FortML's deterministic Park--Miller "
        "bootstrap stream and weighted exhaustive-split CART policy in an "
        "independent NumPy oracle. It checks scalar and multi-output predictions, "
        "staged prefixes, split-frequency feature importance, the fixed-state "
        "zero JVP, and the typed CUDA refusal.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Compiler/flags: `{details['compiler']} {details['flags']}`\n"
        f"- Release wall time: `{wall_seconds:.6g}` s\n"
        f"- Maximum prediction oracle error: `{prediction_error:.6g}`\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n"
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
