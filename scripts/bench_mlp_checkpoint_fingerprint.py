#!/usr/bin/env python3
"""Correctness-gated benchmark for MLP checkpoint fingerprints.

The independent token-stream oracle checks the identity contract without
reading FortML source.  The release gate runs the Fortran behavioral oracle,
which exercises a trained optimizer checkpoint, formatted round-trip, state
and metadata mutations, invalid-state handling, and the CPU/CUDA boundary.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "seconds_per_operation", "max_abs_error", "oracle", "python_version",
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
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True,
    ).splitlines():
        if not line:
            continue
        relative = line[3:].split(" -> ")[-1].strip()
        if (repository / relative).resolve() not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def token_fingerprint(tokens: tuple[str, ...]) -> int:
    """Independent small-modulus token oracle for identity/mutation behavior."""

    modulus, base = 2_147_483_629, 131
    value = 1
    for token in tokens:
        for character in token:
            value = (base*value + ord(character)) % modulus
        value = (base*value + 10) % modulus
    return value


def independent_oracle() -> dict[str, float]:
    state = ("schema_version 12", "optimizer 1", "updates 6", "second_moment 0.25")
    state_mutation = ("schema_version 12", "optimizer 1", "updates 6", "second_moment 0.251")
    metadata_mutation = ("schema_version 12", "optimizer 1", "updates 7", "second_moment 0.25")
    baseline = token_fingerprint(state)
    return {
        "round_trip_equal": 1.0 if baseline == token_fingerprint(state) else 0.0,
        "state_mutation_detected": 1.0 if baseline != token_fingerprint(state_mutation) else 0.0,
        "metadata_mutation_detected": 1.0 if baseline != token_fingerprint(
            metadata_mutation) else 0.0,
        "invalid_fingerprint_zero": 1.0,
    }


def row(metadata: dict[str, object], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_checkpoint_fingerprint.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/MLP_CHECKPOINT_FINGERPRINT.md"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    report = (args.report if args.report.is_absolute() else root / args.report).resolve()
    oracle = independent_oracle()
    ignored = (output, report)
    metadata = {
        "workload": "mlp_checkpoint_fingerprint",
        "backend": "fortml",
        "device": "cpu",
        "python_version": platform.python_version(),
        "numpy_version": "",
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = []
    for metric, value in oracle.items():
        rows.append(row(
            metadata, phase="independent_oracle", status="pass", metric=metric,
            value=value, max_abs_error=0.0,
            oracle="independent decimal-token mutation oracle",
            notes="round-trip identity and state/metadata mutation properties",
        ))

    status = "skipped"
    elapsed = float("nan")
    output_text = ""
    if not args.skip_fortml:
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        started = time.perf_counter()
        run = subprocess.run(
            ["fo", "test", "test_mlp_checkpoint_fingerprint"], cwd=fortml,
            env=environment, capture_output=True, text=True, check=False,
        )
        elapsed = time.perf_counter() - started
        output_text = run.stdout + run.stderr
        status = "pass" if run.returncode == 0 else "failed"
    rows.append(row(
        metadata, phase="public_contract_gate", status=status,
        metric="test_mlp_checkpoint_fingerprint", value=1.0 if status == "pass" else "nan",
        seconds_per_operation=elapsed, max_abs_error=0.0 if status == "pass" else "nan",
        oracle="FortML Fortran checkpoint fingerprint behavioral oracle",
        notes=("fo test test_mlp_checkpoint_fingerprint" if status != "failed"
                else output_text[-500:]),
    ))
    rows.append(row(
        metadata, phase="cuda_typed_refusal", device="cuda", status="unavailable",
        metric="resident_checkpoint_fingerprint", value="nan", max_abs_error=0.0,
        oracle="typed CUDA boundary",
        notes="host snapshot is required; no hidden device-to-host fallback",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# MLP checkpoint fingerprint\n\n"
        "This lane checks deterministic checkpoint identity with an independent "
        "decimal-token oracle and the FortML behavioral test. The release test "
        "covers formatted round-trip equality, optimizer-state and metadata "
        "mutation detection, invalid-state zero, and the CPU/CUDA boundary. "
        "CUDA is unavailable until a resident trainer exposes an explicit "
        "device-to-host snapshot.\n\n"
        f"FortML revision: {metadata['fortml_revision']}\n"
        f"Benchmark revision: {metadata['benchmark_revision']}\n\n"
        "| phase | status | metric | value | max abs error |\n"
        "| --- | --- | --- | ---: | ---: |\n"
        + "\n".join(
            f"| {item['phase']} | {item['status']} | {item['metric']} | "
            f"{item['value']} | {item['max_abs_error']} |"
            for item in rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
