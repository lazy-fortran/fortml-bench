# Transactional pipeline input schemas

This lane checks the bounded dense-schema contract added to FortML's
horizontal, sequential, and fan-out basis pipelines. The fixture installs
three unique names (`time`, `position`, `velocity`), validates a matching
candidate, and independently verifies that mismatched or duplicate names are
refused without mutating the installed schema. The Fortran application times
10,000 successful validations before transforming one row through a polynomial
and Fourier feature union.

Run from this repository:

```bash
python -B scripts/bench_pipeline_schema.py \
  --fortml ../fortml --output results/pipeline_schema.csv
```

## Results

| Phase | Backend/device | Result | Evidence |
| --- | --- | --- | --- |
| Independent names/count/refusal oracle | NumPy | Pass | Three unique names, duplicate refusal, mismatch refusal |
| Repeated validation | FortML / CPU | Pass | 10,000/10,000 validations; 8.52e-8 s/op |
| Resident schema validation | FortML / CUDA | Unavailable | Typed `FORTNUM_NOT_IMPLEMENTED`; no host fallback |

The raw rows pin FortML revision
`104d93cf03da057ed53dc6c33e2ccd0b035f01f6`, benchmark revision
`a30027bc6a06b5e7bb89243b0bdbc051171c3576`, GNU Fortran `-O2`, Python 3.14.6,
and NumPy 2.5.1. The timing covers metadata validation only, not numerical
transform throughput or GPU execution.

The schema object currently covers dense column names and count validation.
Dtypes, sparse layouts, estimator-wide metadata routing, and train-only fit
semantics remain explicit roadmap boundaries.
