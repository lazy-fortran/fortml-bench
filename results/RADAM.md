# RAdam optimizer and MLP training benchmark

This lane checks the deterministic Rectified Adam (RAdam) recurrence against
an independent NumPy oracle before retaining a FortML timing. The flat state
workload uses 4,096 parameters for 128 updates with learning rate `1e-2`,
`beta1=0.9`, `beta2=0.99`, and `epsilon=1e-8`. The MLP workload trains a
one-feature linear MLP for 32 full-batch epochs with learning rate `0.08`,
`beta1=0.85`, `beta2=0.95`, and `epsilon=1e-5`. Both oracle paths implement
the `rho_t <= 4` first-moment branch and the rectified variance branch.

The release app is `fortml_bench_radam_training`; the script is
`scripts/bench_radam_training.py`. The NumPy rows are independent CPU
behavioral oracles. FortML CPU rows are retained only when the app output
matches the oracle within `3e-11` absolute error. CUDA rows are explicit
`unavailable`: no resident RAdam state kernel is linked, and the harness never
relabels host execution as device evidence.

```bash
python3 -B scripts/bench_radam_training.py \
  --fortml ../fortml --output results/radam.csv
```

`test_mlp_radam` additionally checks invalid beta/epsilon options, exact
in-memory and formatted checkpoint continuation (format 8/text schema 6),
and `radam_t%step_device`'s output-preserving CUDA refusal. Optimizer-trajectory
hypergradients, FortOpt RAdam adapters, and resident GPU state are intentionally
not claimed by this bounded lane.
