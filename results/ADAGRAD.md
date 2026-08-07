# Adagrad recurrence benchmark

This lane is an independent optimizer-state oracle for the newly available
FortOpt-backed Adagrad path. It uses a 4096-parameter quadratic objective for
128 deterministic steps with learning rate `1e-2` and epsilon `1e-8`, checks

\[
G_t = G_{t-1} + g_t^2, \qquad
\theta_t = \theta_{t-1} - \eta g_t/(\sqrt{G_t}+\epsilon),
\]

and compares an uninterrupted trajectory with a split-at-step-64 checkpoint
and resume. The NumPy lane is timed only after both state and recurrence
checks pass.

FortML currently has no dedicated Adagrad release benchmark app. Its row is
therefore recorded as `unavailable` for the target
`fortml_bench_adagrad_training`; no Adam, AdamW, or CPU timing is relabeled as
Adagrad evidence. Once a release app exports its packed parameters and
accumulator, the same recurrence and resume oracle can gate a FortML timing.

Run:

```bash
python3 scripts/bench_adagrad.py --fortml ../fortml --output results/adagrad.csv
```

The CSV includes exact state error, final parameter norm, provenance, and the
explicit release-target boundary.
