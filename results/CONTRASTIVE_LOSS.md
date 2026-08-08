# Pairwise contrastive loss benchmark

`bench_contrastive_loss.py` checks FortML's pairwise metric-learning loss
against an independent NumPy implementation before retaining timings. The
fixture contains 128 pairs of 16-dimensional embeddings, deterministic
matching/non-matching labels, and positive row weights. It verifies weighted
contrastive value, JVP, VJP, and HVP checksums and records the typed CUDA value
boundary.

Run it from the benchmark checkout with a clean sibling FortML build:

```bash
python -B scripts/bench_contrastive_loss.py \
  --fortml ../fortml --output results/contrastive_loss.csv
```

The recorded CPU rows use 512 repetitions. The independent NumPy checksum
errors are below `3e-16` for value/JVP/VJP and exactly zero for the aggregate
HVP in the checked fixture. CPU seconds are machine-dependent; the CSV retains
the measured seconds per operation plus compiler/source revisions. FortML
reports `FORTNUM_NOT_IMPLEMENTED` for the CUDA request because a resident
pair-distance/reduction kernel is not linked. This is a capability record, not
a host-fallback timing.

The production derivative contract refuses non-matching zero distances and
exact margin boundaries transactionally. These are represented by separate
behavioral-oracle checks in `fortml/test/test_contrastive_loss.f90`.
