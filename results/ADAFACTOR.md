# Adafactor trainer lane

This lane covers FortML's deterministic unfactored Adafactor mode for flat
objective and dense-MLP parameter vectors. The independent NumPy oracle uses

\[
v_t = \rho v_{t-1} + (1-\rho)g_t^2,\qquad
r_t = \max(1,\operatorname{RMS}(v_t)/c),\qquad
x_t = x_{t-1} - \eta g_t/(r_t(\sqrt{v_t}+\epsilon)).
\]

It compares an uninterrupted trajectory with a split trajectory carrying the
second-moment vector and step count, then runs `test_trainer` and
`test_mlp_adafactor` as the public-contract gate. The CUDA row is a typed
unavailable result: the current trainer owns a CPU flat vector and has no
matrix-layout metadata or resident CUDA Adafactor kernel. The benchmark does
not infer factored row/column state from a packed parameter vector.

Run from this repository with:

```bash
python -B scripts/bench_adafactor.py --fortml ../fortml \
  --output results/adafactor.csv
```

The checked-in CSV records the independent recurrence, public test gate, and
device boundary together with both repository revisions.
