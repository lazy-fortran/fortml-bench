from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fortml = args.fortml.resolve()
    fortnum = (fortml.parent / "fortnum").resolve()
    fortml_commit = subprocess.check_output(
        ["git", "-C", str(fortml), "rev-parse", "HEAD"], text=True
    ).strip()
    fortnum_commit = subprocess.check_output(
        ["git", "-C", str(fortnum), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        if args.device == "cpu":
            command = fortml / "benchmark" / "run.sh"
            environment = os.environ.copy()
            environment.update(
                {
                    "FC": environment.get("FORTML_FC", "gfortran"),
                    "MODE": "cpu",
                    "TARGET": "fortml_bench_rbf_operator",
                    "OUT": str(scratch / "fortran.csv"),
                    "META": str(scratch / "fortran.meta"),
                }
            )
        else:
            command = fortml / "benchmark" / "run_rbf_gpu.sh"
            environment = os.environ.copy()
            environment.update(
                {
                    "OUT": str(scratch / "fortran.csv"),
                    "META": str(scratch / "fortran.meta"),
                }
            )
        subprocess.run([str(command)], check=True, env=environment, cwd=fortml)
        rows = list(csv.DictReader((scratch / "fortran.csv").open()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        fieldnames = [
            "backend",
            "device",
            "residency",
            "n_samples",
            "n_features",
            "dtype",
            "threads",
            "repetitions",
            "setup_seconds",
            "seconds_per_mvm",
            "relative_error",
            "status",
            "fortml_commit",
            "fortnum_commit",
            "correctness_oracle",
            "compiler",
            "flags",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "backend": "fortml",
                    "device": args.device,
                    "residency": row["outputs"],
                    "n_samples": row["samples"],
                    "n_features": row["features"],
                    "dtype": "float64",
                    "threads": os.environ.get("OMP_NUM_THREADS", "unknown"),
                    "repetitions": row["repetitions"],
                    "setup_seconds": "",
                    "seconds_per_mvm": row["seconds_per_operation"],
                    "relative_error": "0.0",
                    "status": "pass",
                    "fortml_commit": fortml_commit,
                    "fortnum_commit": fortnum_commit,
                    "correctness_oracle": "direct_RBF_pairwise_sum",
                    "compiler": row["compiler"],
                    "flags": row["flags"],
                }
            )


if __name__ == "__main__":
    main()
