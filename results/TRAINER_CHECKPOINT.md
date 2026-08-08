# Generic trainer checkpoint benchmark

`bench_trainer_checkpoint.py` uses an independent NumPy Adam recurrence and a
separate weighted quadratic Lion recurrence. It compares uninterrupted and
split continuations, including parameters, optimizer state, and step counts.
`test_trainer` also checks EMA/history state and rejects truncated or extra
text records transactionally for the generic trainer.

Run:

```bash
python3 -B scripts/bench_trainer_checkpoint.py \
  --fortml ../fortml --output results/trainer_checkpoint.csv
```

Checkpoint state is host-resident; the CSV therefore retains an explicit CUDA
`unavailable` row rather than implying accelerator support.
