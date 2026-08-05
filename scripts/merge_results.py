from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    fieldnames = []
    for input_path in args.inputs:
        if not input_path.exists():
            continue
        with input_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            for fieldname in reader.fieldnames or ():
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            rows.extend(reader)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
