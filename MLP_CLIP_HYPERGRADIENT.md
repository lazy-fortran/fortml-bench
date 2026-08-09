# MLP global-norm clipping hypergradients

This lane measures a four-step full-batch SGD trajectory with coupled L2 and a
global gradient-clip threshold. The optimized coordinates are
`log(learning_rate)`, `log(l2)`, and `log(gradient_clip_norm)`.

The correctness gate uses an independent NumPy recurrence for a two-parameter
linear MLP. NumPy applies clipping after the coupled-L2 gradient, evaluates the
held-out MSE, and central-differences all three outer coordinates plus one
direction. The FortML release application must emit the complete value,
gradient, and JVP array before its timing is retained. The CSV records the
largest error over that array in every FortML pass row.

The CPU implementation differentiates the clipping scale on a fixed active
set. It returns a typed status at the exact clipping kink. Outer HVPs also have
a typed unavailable status because they require third network derivatives.
The CUDA rows are unavailable until the resident MLP gradient, clipping, and
hypergradient state share one device graph.

Run the benchmark from this repository:

```bash
python -B scripts/bench_mlp_clip_hypergradient.py \
    --fortml ../fortml \
    --output results/mlp_clip_hypergradient.csv
```

The result file records Python, NumPy, compiler, FortML, and benchmark
revisions. A `+dirty` suffix excludes a row from release evidence.
