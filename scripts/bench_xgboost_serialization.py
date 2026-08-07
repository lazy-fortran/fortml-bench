#!/usr/bin/env python3
"""Correctness-gated XGBoost text persistence benchmark.

The fixture emits a fitted model before and after save/load.  In addition to
checking byte-preserving predictions, this script parses FortML's text schema
and independently walks every serialized tree in NumPy.  No FortML prediction
is used as the semantic oracle.
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


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "seconds", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


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
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture() -> np.ndarray:
    query = np.empty((7, 2), dtype=np.float64)
    query[:, 0] = (-1.0, 0.0, 1.5, 3.5, 5.0, 7.0, 9.0)
    query[:, 1] = (-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, 1.0)
    return query


def build_probe(fortml: Path, fixture_path: Path) -> tuple[str, str, float]:
    build = subprocess.run(["fo", "build", "--flag", "-O2"], cwd=fortml,
                           capture_output=True, text=True, check=False)
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
    with tempfile.TemporaryDirectory(prefix="fortml-xgb-serialization-", dir="/mnt/storage") as directory:
        directory_path = Path(directory)
        source = directory_path / fixture_path.name
        executable = directory_path / "xgboost_serialization_probe"
        model_path = directory_path / "model.txt"
        source.write_bytes(fixture_path.read_bytes())
        command = compiler + ["-O2", "-ffree-line-length-none", "-I", str(module_dir),
                              str(source), str(archive), "-o", str(executable)]
        link = subprocess.run(command, cwd=fortml, capture_output=True, text=True, check=False)
        if link.returncode:
            raise RuntimeError(link.stderr.strip() or link.stdout.strip())
        started = time.perf_counter()
        run = subprocess.run([str(executable), str(model_path)], capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        if run.returncode:
            raise RuntimeError(run.stderr.strip() or run.stdout.strip())
        return run.stdout, model_path.read_text(), elapsed


def parse_probe(stdout: str) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, float | int] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or not fields[0].startswith("xgb_serialization_"):
            continue
        key = fields[0]
        if key in {"xgb_serialization_prediction_before", "xgb_serialization_prediction_after",
                   "xgb_serialization_margin_before", "xgb_serialization_margin_after"}:
            name = key.removeprefix("xgb_serialization_")
            arrays.setdefault(name, np.zeros(7))[int(fields[1]) - 1] = float(fields[2].replace("D", "E"))
        elif key in {"xgb_serialization_staged_before", "xgb_serialization_staged_after"}:
            name = key.removeprefix("xgb_serialization_")
            arrays.setdefault(name, np.zeros((7, 4)))[int(fields[1]) - 1, int(fields[2]) - 1] = float(fields[3].replace("D", "E"))
        elif key == "xgb_serialization_cuda":
            metadata["cuda"] = int(fields[1])
        elif key == "xgb_serialization_estimator_count":
            metadata["estimator_count"] = int(fields[1])
        elif key == "xgb_serialization_best_iteration":
            metadata["best_iteration"] = int(fields[1])
        elif key == "xgb_serialization_best_loss":
            metadata["best_loss"] = float(fields[1].replace("D", "E"))
        elif key.startswith("xgb_serialization_monotone_"):
            metadata[key.removeprefix("xgb_serialization_")] = int(fields[1])
    return {**arrays, **metadata}


def read_record(lines: list[str], cursor: list[int], expected: str, kind: str = "float") -> float | int:
    if cursor[0] >= len(lines):
        raise RuntimeError(f"truncated serialized model at {expected}")
    fields = lines[cursor[0]].split(); cursor[0] += 1
    if len(fields) != 2 or fields[0] != expected:
        raise RuntimeError(f"expected {expected!r}, found {lines[cursor[0] - 1]!r}")
    return int(fields[1]) if kind == "int" else float(fields[1].replace("D", "E"))


def parse_model(text: str) -> dict[str, object]:
    lines = [line.strip() for line in text.splitlines()]
    if not lines or lines[0] != "FORTML_XGBOOST_TEXT":
        raise RuntimeError("serialized model magic mismatch")
    cursor = [1]
    schema = read_record(lines, cursor, "schema_version", "int")
    if schema != 1:
        raise RuntimeError(f"unsupported serialized schema {schema}")
    n_inputs = int(read_record(lines, cursor, "n_inputs", "int"))
    n_estimators = int(read_record(lines, cursor, "n_estimators", "int"))
    requested = int(read_record(lines, cursor, "requested_estimators", "int"))
    scalar_names = ("objective_code", "tree_method_code", "max_bin", "max_depth", "min_samples_leaf",
                    "early_stopping_rounds")
    scalars: dict[str, float | int] = {name: read_record(lines, cursor, name, "int") for name in scalar_names}
    for name in ("learning_rate", "base_score", "objective_parameter", "l1", "l2", "gamma",
                 "min_child_weight", "early_stopping_min_delta", "subsample", "colsample_bytree"):
        scalars[name] = read_record(lines, cursor, name)
    scalars["seed"] = read_record(lines, cursor, "seed", "int")
    scalars["restore_best"] = read_record(lines, cursor, "restore_best", "int")
    scalars["missing_code"] = read_record(lines, cursor, "missing_code", "int")
    scalars["best_iteration"] = read_record(lines, cursor, "best_iteration", "int")
    scalars["best_validation_loss"] = read_record(lines, cursor, "best_validation_loss")
    scalars["early_stopped"] = read_record(lines, cursor, "early_stopped", "int")
    monotone_count = int(read_record(lines, cursor, "monotone_count", "int"))
    monotone = [int(read_record(lines, cursor, "monotone_item", "int")) for _ in range(monotone_count)]
    tree_count = int(read_record(lines, cursor, "tree_count", "int"))
    trees: list[dict[str, object]] = []
    for tree_number in range(1, tree_count + 1):
        if int(read_record(lines, cursor, "tree_begin", "int")) != tree_number:
            raise RuntimeError("tree numbering mismatch")
        tree: dict[str, object] = {"n_nodes": int(read_record(lines, cursor, "n_nodes", "int"))}
        for name in ("depth", "feature_index", "left_count", "right_count"):
            tree[name] = read_record(lines, cursor, name, "int")
        for name in ("threshold", "left_weight", "right_weight", "split_gain"):
            tree[name] = read_record(lines, cursor, name)
        tree["has_split"] = read_record(lines, cursor, "has_split", "int")
        node_count = int(read_record(lines, cursor, "node_count", "int"))
        if node_count != tree["n_nodes"]:
            raise RuntimeError("node count mismatch")
        nodes: list[dict[str, object]] = []
        for node_number in range(1, node_count + 1):
            if int(read_record(lines, cursor, "node_index", "int")) != node_number:
                raise RuntimeError("node numbering mismatch")
            node: dict[str, object] = {}
            for name in ("feature", "left_child", "right_child"):
                node[name] = read_record(lines, cursor, name, "int")
            for name in ("node_threshold", "weight", "node_gain", "node_cover"):
                node[name] = read_record(lines, cursor, name)
            node["leaf"] = read_record(lines, cursor, "leaf", "int")
            node["missing_left"] = read_record(lines, cursor, "missing_left", "int")
            nodes.append(node)
        if cursor[0] >= len(lines) or lines[cursor[0]] != "tree_end":
            raise RuntimeError("tree terminator mismatch")
        cursor[0] += 1
        tree["nodes"] = nodes
        trees.append(tree)
    if cursor[0] >= len(lines) or lines[cursor[0]] != "end":
        raise RuntimeError("serialized model has trailing or missing records")
    if cursor[0] + 1 != len(lines):
        raise RuntimeError("serialized model has records after end")
    return {"n_inputs": n_inputs, "n_estimators": n_estimators, "requested": requested,
            "scalars": scalars, "monotone": monotone, "trees": trees}


def tree_correction(tree: dict[str, object], query: np.ndarray) -> np.ndarray:
    result = np.empty(query.shape[0], dtype=np.float64)
    nodes = tree["nodes"]
    for row, values in enumerate(query):
        node_index = 0
        while not nodes[node_index]["leaf"]:
            node = nodes[node_index]
            feature = int(node["feature"]) - 1
            if np.isnan(values[feature]):
                go_left = bool(node["missing_left"])
            else:
                go_left = values[feature] < float(node["node_threshold"])
            node_index = (int(node["left_child"]) if go_left else int(node["right_child"])) - 1
        result[row] = float(nodes[node_index]["weight"])
    return result


def serialized_oracle(model: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    query = fixture(); scalars = model["scalars"]
    margin = np.full(query.shape[0], float(scalars["base_score"])); staged = np.empty((query.shape[0], model["n_estimators"]))
    for i, tree in enumerate(model["trees"]):
        margin += float(scalars["learning_rate"]) * tree_correction(tree, query)
        staged[:, i] = margin
    return margin, staged


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}; result.update(details); result.update(values); return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/xgboost_serialization.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; fortml = args.fortml.resolve()
    stdout, model_text, elapsed = build_probe(fortml, root / "fixtures" / "xgboost_serialization_probe.f90")
    observed = parse_probe(stdout); model = parse_model(model_text); oracle_margin, oracle_staged = serialized_oracle(model)
    error = max(
        float(np.max(np.abs(observed["prediction_before"] - observed["prediction_after"]))),
        float(np.max(np.abs(observed["margin_before"] - observed["margin_after"]))),
        float(np.max(np.abs(observed["staged_before"] - observed["staged_after"]))),
        float(np.max(np.abs(observed["prediction_before"] - oracle_margin))),
        float(np.max(np.abs(observed["margin_before"] - oracle_margin))),
        float(np.max(np.abs(observed["staged_before"] - oracle_staged))),
    )
    if error > 3.0e-12:
        raise RuntimeError(f"XGBoost serialization/oracle mismatch: {error:.3e}")
    scalars = model["scalars"]
    if (model["n_inputs"], model["n_estimators"], model["requested"], model["monotone"],
        scalars["missing_code"], observed["cuda"], observed["estimator_count"], observed["best_iteration"]) != (2, 4, 4, [1, 0], 1, 3, 4, 2):
        raise RuntimeError("XGBoost serialization metadata/device contract mismatch")
    output = args.output.resolve()
    details = {"oracle": "independent NumPy parser and node-walk of FortML text schema",
               "python_version": platform.python_version(), "numpy_version": np.__version__,
               "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
               "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O2"}
    records = [
        row(details, workload="xgboost_serialization", phase="save_load_roundtrip", backend="fortml",
            device="cpu", status="pass", metric="max_abs_error", value=error, max_abs_error=error,
            seconds=elapsed, notes="predictions, margins, staged outputs, and metadata survive text round trip"),
        row(details, workload="xgboost_serialization", phase="serialized_tree_oracle", backend="fortml",
            device="cpu", status="pass", metric="max_abs_error", value=error, max_abs_error=error,
            seconds=elapsed, notes="independent parser walks four trees, missing-policy and monotone metadata"),
        row(details, workload="xgboost_serialization", phase="device_capability", backend="fortml", device="cuda",
            status="unavailable", metric="resident_tree_prediction", value="nan", max_abs_error="nan",
            oracle="typed device contract", notes="FORTNUM_NOT_IMPLEMENTED=3; no resident CUDA tree kernel"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}; max_abs_error={error:.3e}")


if __name__ == "__main__": main()
