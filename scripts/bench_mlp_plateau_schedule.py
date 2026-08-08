#!/usr/bin/env python3
"""Correctness-gated metric-aware plateau schedule benchmark.

The state transition and derivatives are implemented independently in Python.
The FortML app must emit the complete transition array before its timing is
retained.  Integer patience decisions and comparison derivatives are checked
as explicit active-set contracts.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


BASE_RATE = 0.2
PATIENCE = 2
MIN_DELTA = 0.05
FACTOR = 0.5
METRICS = ((1.1, 1.0, 0.99, 0.90, 0.90, 0.90),
           (0.8, 0.7, 0.7, 0.85, 0.85, 0.85))
MODES = ("minimize", "maximize")
INITIAL_BEST = (1.0, 0.7)
QUANTITIES = ("rate", "d_base_rate", "d_metric", "d_best_metric", "d_min_delta",
              "d_factor", "next_best_metric", "next_bad_updates", "next_reductions",
              "improved", "reduced")
FIELDS = ("workload", "scenario", "phase", "backend", "device", "status", "index",
          "value", "max_abs_error", "oracle", "python_version", "fortml_revision",
          "benchmark_revision", "compiler", "flags", "notes")


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                   text=True).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = any(line[3:].split(" -> ")[-1].strip() not in ignored_names
                for line in subprocess.check_output(
                    ["git", "-C", str(repository), "status", "--porcelain"],
                    text=True).splitlines())
    return head + ("+dirty" if dirty else "")


def step(mode: str, metric: float, best: float, bad: int, reductions: int,
         factor: float = FACTOR) -> tuple[dict[str, float], float, int, int]:
    improved = ((metric < best - MIN_DELTA) if mode == "minimize"
                else (metric > best + MIN_DELTA))
    next_best = metric if improved else best
    if improved:
        next_bad, next_reductions, reduced = 0, reductions, False
    elif bad >= PATIENCE - 1:
        next_bad, next_reductions, reduced = 0, reductions + 1, True
    else:
        next_bad, next_reductions, reduced = bad + 1, reductions, False
    power = factor ** next_reductions
    values = {
        "rate": BASE_RATE * power,
        "d_base_rate": power,
        "d_metric": 0.0,
        "d_best_metric": 0.0,
        "d_min_delta": 0.0,
        "d_factor": (BASE_RATE * next_reductions * factor ** (next_reductions - 1)
                      if next_reductions else 0.0),
        "next_best_metric": next_best,
        "next_bad_updates": float(next_bad),
        "next_reductions": float(next_reductions),
        "improved": float(improved),
        "reduced": float(reduced),
    }
    return values, next_best, next_bad, next_reductions


def independent_oracle() -> dict[tuple[str, int, str], float]:
    expected: dict[tuple[str, int, str], float] = {}
    h = 1.0e-6
    for scenario, mode in enumerate(MODES):
        best, bad, reductions = INITIAL_BEST[scenario], 0, 0
        for observation, metric in enumerate(METRICS[scenario], 1):
            values, next_best, next_bad, next_reductions = step(
                mode, metric, best, bad, reductions)
            index = 100 * scenario + observation
            for quantity, value in values.items():
                expected[(mode, index, quantity)] = value
            plus = step(mode, metric, best, bad, reductions, FACTOR + h)[0]["rate"]
            minus = step(mode, metric, best, bad, reductions, FACTOR - h)[0]["rate"]
            if abs(values["d_factor"] - (plus - minus) / (2.0 * h)) > 2.0e-10:
                raise RuntimeError(f"factor oracle failed for {mode} observation {observation}")
            best, bad, reductions = next_best, next_bad, next_reductions
    return expected


def row(details: dict[str, str], scenario: str, phase: str, backend: str,
        status: str, index: int | str, value: float | str, error: float | str,
        notes: str, device: str = "cpu") -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update({"workload": "mlp_plateau_schedule", "scenario": scenario,
                   "phase": phase, "backend": backend, "device": device,
                   "status": status, "index": index, "value": value,
                   "max_abs_error": error,
                   "oracle": "independent Python plateau transition and FD factor",
                   "notes": notes})
    return result


def run_numpy(details: dict[str, str], expected: dict[tuple[str, int, str], float]) -> list[dict[str, object]]:
    rows = []
    started = time.perf_counter()
    for key, value in expected.items():
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite Python oracle value {key}")
    elapsed = (time.perf_counter() - started) / max(len(expected), 1)
    for (scenario, index, quantity), value in expected.items():
        rows.append(row(details, scenario, quantity, "python_oracle", "pass", index,
                        value, 0.0, f"active-set transition and factor FD checked; {elapsed:.3e}s"))
    return rows


def parse_oracle(path: Path) -> dict[tuple[str, int, str], float]:
    actual = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            key = (record["quantity"], int(record["index"]), "")
            actual[(record["quantity"], int(record["index"]), "")] = float(record["value"])
    return actual


def run_fortml(fortml: Path, details: dict[str, str],
               expected: dict[tuple[str, int, str], float]) -> list[dict[str, object]]:
    fixture = fortml / "app" / "fortml_bench_mlp_plateau_schedule.f90"
    if not fixture.is_file():
        return [row(details, "all", "release_app", "fortml", "unavailable", "", "", "",
                    "release target source is absent")]
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return [row(details, "all", "release_app", "fortml", "unavailable", "", "", "",
                    "fo build failed")]
    archives = list((fortml / "build" / "fo" / "lib").glob("*.a"))
    if not archives:
        return [row(details, "all", "release_app", "fortml", "unavailable", "", "", "",
                    "fo build produced no archive")]
    compiler = shlex.split(environment["FO_FC"])
    if not compiler or shutil.which(compiler[0]) is None:
        return [row(details, "all", "release_app", "fortml", "unavailable", "", "", "",
                    "Fortran compiler unavailable")]
    archive = max(archives, key=lambda path: path.stat().st_mtime_ns)
    with tempfile.TemporaryDirectory(prefix="fortml-plateau-", dir="/mnt/storage") as directory:
        directory_path = Path(directory)
        executable = directory_path / "plateau_schedule_probe"
        link = subprocess.run(
            compiler + ["-O3", "-ffree-line-length-none", "-I", str(fortml / "build" / "fo" / "mod"),
                        str(fixture), str(archive), "-o", str(executable)],
            cwd=fortml, capture_output=True, text=True)
        if link.returncode != 0:
            raise RuntimeError(link.stderr.strip() or link.stdout.strip())
        oracle_path = directory_path / "oracle.csv"
        check_environment = dict(environment)
        check_environment.update({"FORTML_BENCH_MLP_PLATEAU_ORACLE": str(oracle_path),
                                   "FORTML_BENCH_ORACLE_ONLY": "1"})
        check = subprocess.run([str(executable)], cwd=fortml, env=check_environment,
                               capture_output=True, text=True)
        if check.returncode != 0 or not oracle_path.is_file():
            raise RuntimeError(check.stderr.strip() or "FortML plateau app emitted no oracle")
        actual_raw = {}
        with oracle_path.open(newline="") as stream:
            for record in csv.DictReader(stream):
                actual_raw[(record["quantity"], int(record["index"]))] = float(record["value"])
        required = {(quantity, index) for _, index, quantity in expected}
        if set(actual_raw) != required:
            raise RuntimeError("FortML plateau app omitted a complete transition array")
        errors = [abs(actual_raw[(quantity, index)] - value)
                  for (_, index, quantity), value in expected.items()]
        error = max(errors)
        if error > 2.0e-12:
            raise RuntimeError(f"FortML plateau oracle mismatch: {error:.3e}")
        timed = subprocess.run([str(executable)], cwd=fortml, env=environment,
                               capture_output=True, text=True)
        if timed.returncode != 0:
            raise RuntimeError(timed.stderr.strip() or "FortML plateau timing failed")
        timing = next((float(line.split(",", 1)[1]) for line in timed.stdout.splitlines()
                       if line.startswith("mlp_plateau_schedule_rate_with_metric,")), None)
        if timing is None:
            raise RuntimeError("FortML plateau app emitted no timing")
    return [row(details, scenario, quantity, "fortml", "pass", index,
                 actual_raw[(quantity, index)],
                 abs(actual_raw[(quantity, index)] - value),
                 f"complete transition array checked before timing; {timing:.3e}s/op")
            for scenario, index, quantity in expected for value in [expected[(scenario, index, quantity)]]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_plateau_schedule.csv"))
    args = parser.parse_args()
    root, fortml, output = Path(__file__).resolve().parents[1], args.fortml.resolve(), args.output.resolve()
    details = {"python_version": platform.python_version(),
               "fortml_revision": revision(fortml),
               "benchmark_revision": revision(root, (output,)),
               "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3"}
    expected = independent_oracle()
    rows = run_numpy(details, expected)
    rows.extend(run_fortml(fortml, details, expected))
    rows.extend(row(details, mode, "device_capability", "fortml", "unavailable", "", "", "",
                    "schedule device_supported(CUDA)=false; no resident metric trainer lowering",
                    device="cuda") for mode in MODES)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
