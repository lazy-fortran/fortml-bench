# MLP schedule products

This lane checks the typed `fortml_mlp_schedules` contract for six schedule
families: constant, linear warm-up, cosine decay, warm-up plus cosine,
exponential decay, and one-cycle. The fixture uses base rate `0.2` and
one-based updates `1, 2, 5, 10, 12`. For every family/update pair it records
the rate and analytic products with respect to the base rate, minimum-rate
fraction, and decay factor; one-cycle also records peak-rate and final-rate
fraction products.

The independent NumPy oracle reconstructs each formula and checks all used
continuous products by central differences before timing. The FortML release
app must emit the complete 180-value array (`6 * 5 * 6`) before any timing is
retained. Missing or malformed release output is recorded as `unavailable`,
never relabelled as a NumPy or GPU result.

Run:

```bash
python3 scripts/bench_mlp_schedules.py \
  --fortml ../fortml --output results/mlp_schedules.csv
```

The app reports a resident CPU scalar recurrence time after its complete-array
oracle pass. The current CSV has 216 passing CPU rows and six explicit CUDA
`unavailable` capability rows. This is a schedule-product gate, not an
end-to-end neural-network training benchmark and does not imply resident CUDA
or OpenACC execution.
