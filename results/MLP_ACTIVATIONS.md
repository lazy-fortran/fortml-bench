# Dense MLP activation benchmark

`scripts/bench_mlp_activations.py` evaluates one deterministic `8-32-4`
dense network on 2,048 samples for each activation exposed by `fortml_mlp`:
linear, `tanh`, ReLU, tanh-approximate GELU, SiLU, ELU, softplus, and fixed
slope (`0.01`) leaky ReLU. The NumPy implementation computes an independent
packed-weight checksum before any FortML timing is retained. The raw record is
[`mlp_activations.csv`](mlp_activations.csv).

The current record contains 30 rows: ten NumPy oracle passes, ten FortML CPU
passes, and ten explicit CUDA `unavailable` capability rows. The largest
FortML/NumPy checksum error is `2.51e-12`; all rows pass the `2e-11` gate. CPU
timings are per 2,048-sample forward call and include no process-startup time:

| activation | FortML CPU seconds | max absolute checksum error |
| --- | ---: | ---: |
| linear | 1.0460e-3 | 1.14e-13 |
| tanh | 1.7318e-3 | 1.82e-12 |
| relu | 1.0309e-3 | 7.96e-13 |
| gelu | 1.7417e-3 | 2.50e-12 |
| silu | 1.2182e-3 | 2.84e-13 |
| elu | 1.1133e-3 | 3.41e-13 |
| softplus | 1.4856e-3 | 3.84e-13 |
| leaky_relu | 1.0658e-3 | 9.09e-13 |

Reproduce with:

```bash
.venv/bin/python -B scripts/bench_mlp_activations.py \
    --fortml ../fortml --output results/mlp_activations.csv
```

The activation values are used by the MLP JVP, VJP, and HVP products; their
independent derivative checks live in `fortml/test/test_mlp_activations.f90`.
The CUDA rows are intentionally untimed: the high-level MLP forward and
gradient path has no resident activation kernel in this release, so the
benchmark does not relabel host work as device evidence.
