# Benchmark result schema v1

Every release benchmark is a CSV of observations. The header identifies the
workload and phase, the execution backend and device, the correctness status,
the measured metric, and the provenance needed to reproduce the row.

The v1 required columns are:

`workload, phase, backend, device, status, metric, value, max_abs_error,
oracle, fortml_revision, benchmark_revision, compiler, flags, notes`.

The optional columns record workload shape and measurement detail:
`variant, dimensions, n_samples, n_features, n_outputs, n_classes,
n_train, n_validation, epochs, batch_size, evaluations, repetitions,
compile_seconds, warmup_seconds, train_seconds, predict_seconds,
seconds_per_operation, peak_host_bytes, peak_device_bytes, transfer_bytes,
warmup_iterations, seed, hardware, python_version, numpy_version,
scipy_version, fortad_revision, fortsym_revision, fortopt_revision`.

`status` is one of `pass`, `failed`, `skipped`, `unavailable`, `refused`, or
`conditional`. A passing row has a finite `max_abs_error` and names an
independent `oracle`. An unavailable or refused row records the capability
boundary in `notes` and does not claim a host fallback. Revision fields are
full Git object names and cannot contain `+dirty` in release evidence.

Rows separate correctness from performance. `seconds_per_operation` excludes
compile and warmup time when the harness can measure those phases. Device
memory and transfer fields are optional until the backend exposes counters,
but a missing counter must remain empty rather than being reported as zero.

Validate release rows with:

```bash
python -B scripts/validate_result_schema.py \
  results/mlp_minibatch_adam_hypergradient.csv \
  results/gp_ordinal_cutpoints.csv \
  results/boosted_partial_dependence.csv
```

The validator also supports `--all` for an audit of historical rows. The
2026-08-09 inventory contains 329 tracked CSVs: 184 pass the v1 release
schema and 145 remain legacy/non-release records. Those legacy files are kept
as historical research data, but they are not release evidence and are not
allowed to silently pass the release gate. See
[`RESULTS_INVENTORY.md`](RESULTS_INVENTORY.md) for the audit policy and the
exact commands used to quarantine or rerun a row.
