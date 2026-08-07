# Integer one-hot encoder workload

`scripts/bench_one_hot_encoder.py` writes
[`one_hot_encoder.csv`](one_hot_encoder.csv).  The fixture contains 512
training rows and five integer categorical columns.  Every column includes an
explicit missing sentinel (`-99`), categories are sorted, unknown values are
ignored, missing values are represented as a fitted category, and the first
category in each block is dropped.  A separate 128-row query exercises both
missing values and an unknown code.

The NumPy implementation is an independent behavioral oracle.  It checks the
complete sorted category lists, packed category/output offsets, and every
dense transformed element before timing fit and transform.  The optional
scikit-learn row uses
`OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)`
(with the legacy `sparse=False` spelling when needed) and must agree element
for element.

Run the lane serially:

```bash
.venv/bin/python -B scripts/bench_one_hot_encoder.py \
  --fortml ../fortml --output results/one_hot_encoder.csv
```

Integer categorical data has no canonical tangent or cotangent space.  NumPy,
scikit-learn, and FortML therefore record JVP/VJP as explicit `refused` rows;
the implementation never reports a misleading zero derivative.

The intended FortML release target is `fortml_bench_one_hot_encoder`.  It
should read `FORTML_BENCH_ONE_HOT_ORACLE` and write one-based CSV quantities
`category`, `category_offset`, `output_offset`, and `transformed`, while
emitting `one_hot_fit` and `one_hot_transform` timing records.  The harness
checks every category, offset, and output element.  An absent target or
incomplete output remains an explicit `unavailable` record.  This lane is
CPU-only and does not imply device-resident categorical support.
