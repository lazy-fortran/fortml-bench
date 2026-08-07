# RMSprop trajectory hypergradient benchmark

This lane checks the exact fixed full-batch RMSprop trajectory objective against
an independent NumPy implementation. The packed outer variable is

```text
[ log_learning_rate, log_l2, decay, log_epsilon, momentum ]
```

The NumPy oracle evaluates the two-parameter linear MLP trajectory for four
steps and obtains all five gradient components and the directional JVP by
central finite differences (`h=2e-6`). It records separate centered and
uncentered rows before any release timing is retained. The
centered fixture uses learning rate `0.12`, L2 `0.07`, decay `0.78`, epsilon
`0.03`, momentum `0.21`, and validation rows held out from the inner updates.

The FortML target is `fortml_bench_rmsprop_hypergradient` for the centered
fixture. It exports the
validation value, five packed gradient components, and JVP through
`FORTML_BENCH_RMSPROP_HYPERGRADIENT_ORACLE`, followed by a repeated
`value_gradient` timing. The harness rejects the release timing when any
reported product differs from NumPy by more than `3e-10`. The direct
no-autodiff RMSprop state kernel is checked separately by
`../fortml/test/run_cuda_rmsprop_state.sh`. CUDA rows for the complete MLP
HVP trajectory remain explicit `unavailable` capability boundaries, and no
CPU timing is relabeled as CUDA evidence.

Run:

```bash
.venv/bin/python -B scripts/bench_rmsprop_hypergradient.py \
  --fortml ../fortml --output results/rmsprop_hypergradient.csv
```

The raw CSV records the independent oracle rows, release-app rows, CUDA
refusals, compiler flags, and both repository revisions.
