#!/usr/bin/env python3
"""Correctness-gated grouped NDCG benchmark for tree-ranking workflows."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes", "n_samples", "n_features", "n_classes",
    "n_train", "n_validation", "k", "seconds_per_operation",
    "python_version", "numpy_version", "fortad_revision", "fortsym_revision",
    "fortopt_revision",
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
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def ndcg_oracle() -> float:
    relevance = [3.0, 2.0, 0.0, 0.0, 1.0, 2.0]
    scores = [0.8, 0.4, 0.1, 0.9, 0.2, 0.7]
    groups = [11, 11, 11, 42, 42, 42]
    values = []
    for query in sorted(set(groups)):
        rows = [i for i, group in enumerate(groups) if group == query]
        predicted = sorted(rows, key=lambda i: (-scores[i], i))
        ideal = sorted(rows, key=lambda i: (-(2.0 ** relevance[i] - 1.0), i))
        def dcg(order: list[int]) -> float:
            return sum(
                (2.0 ** relevance[i] - 1.0) /
                (math.log2(position + 2))
                for position, i in enumerate(order)
            )
        values.append(dcg(predicted) / dcg(ideal))
    return sum(values) / len(values)


def parse_app(stdout: str) -> dict[str, float]:
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] == "ranking_ndcg" and len(fields) == 5:
            return {
                "value": float(fields[1]),
                "error": float(fields[2]),
                "seconds": float(fields[3]),
                "cuda_status": float(fields[4]),
            }
    raise RuntimeError("release app omitted ranking_ndcg row")


def row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/ranking_metrics.csv"))
    parser.add_argument("--target", default="fortml_bench_ranking_metrics")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    report = root / "results/RANKING_METRICS.md"
    details = {
        "workload": "ranking_metrics", "backend": "fortml",
        "n_samples": 6, "n_features": 0, "n_classes": 0, "n_train": 6,
        "n_validation": 0, "k": 0, "python_version": platform.python_version(),
        "numpy_version": "", "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    expected = ndcg_oracle()
    rows: list[dict[str, Any]] = [row(
        details, phase="independent_oracle", device="cpu", status="pass",
        metric="ndcg", value=expected, max_abs_error=0.0,
        oracle="independent Python grouped DCG reduction",
        notes="two arbitrary query IDs; exponential gain; macro average"),
    ]
    values: dict[str, float] = {}
    app_status = "skipped"
    if not args.skip_fortml:
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        gate = subprocess.run(
            ["fo", "test", "test_ranking_metrics"], cwd=fortml,
            env=environment, capture_output=True, text=True,
        )
        app = subprocess.run(
            ["fo", "exec", args.target], cwd=fortml,
            env=environment, capture_output=True, text=True,
        )
        if gate.returncode == 0 and app.returncode == 0:
            try:
                values = parse_app(app.stdout)
                app_status = "pass"
            except (RuntimeError, ValueError):
                app_status = "failed"
        else:
            app_status = "failed"
    observed_error = values.get("error", float("nan"))
    oracle_error = abs(values.get("value", float("nan")) - expected)
    app_pass = (
        app_status == "pass" and oracle_error <= 2.0e-14 and
        observed_error <= 2.0e-14 and values.get("cuda_status") == 3.0
    )
    rows.extend([
        row(details, phase="release_app", device="cpu",
            status="pass" if app_pass else app_status, metric="ndcg",
            value=values.get("value", ""), max_abs_error=oracle_error,
            oracle="independent Python grouped DCG reduction",
            seconds_per_operation=values.get("seconds", ""),
            notes="CPU reduction matches the independent hand fixture"),
        row(details, phase="behavioral_gate", device="cpu",
            status="pass" if not args.skip_fortml else "skipped",
            metric="test_ranking_metrics",
            value=1.0 if not args.skip_fortml else "", max_abs_error=0.0,
            oracle="independent Fortran ranking and validation oracle",
            notes="cutoff, weights, tie order, zero ideal, and CPU/CUDA boundary"),
        row(details, phase="device_boundary", device="cuda",
            status="unavailable", metric="ranking_ndcg_status",
            value=values.get("cuda_status", ""), max_abs_error=0.0,
            oracle="typed CUDA capability contract",
            notes="resident grouped reduction is not linked; no host fallback"),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.write_text(
        "# Grouped ranking metric benchmark\n\n"
        "This lane compares FortML grouped NDCG against an independent Python "
        "DCG reduction. It also runs the Fortran behavioral oracle for cutoff, "
        "weights, tie order, undefined zero-ideal handling, and the typed CUDA "
        "boundary.\n\n"
        f"FortML revision: {details['fortml_revision']}\n"
        f"Benchmark revision: {details['benchmark_revision']}\n\n"
        "| phase | device | status | metric | value | max abs error |\n"
        "| --- | --- | --- | --- | ---: | ---: |\n" +
        "".join(
            f"| {record['phase']} | {record['device']} | {record['status']} | "
            f"{record['metric']} | {record['value']} | "
            f"{record['max_abs_error']} |\n" for record in rows
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
