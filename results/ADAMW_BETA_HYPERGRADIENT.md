# AdamW beta-logit hypergradient benchmark

FortML's full AdamW trajectory objective now exposes five packed outer
variables:

\[
(\log \eta,\;\log \lambda_2,\;\log \lambda_d,\;
\operatorname{logit}(\beta_1),\;\operatorname{logit}(\beta_2)).
\]

This lane uses the same independent eight-point fixture as the core
behavioral contract: five training points, three validation points, a
one-weight/one-bias MLP initialized to `[0.15,-0.1]`, four AdamW steps, and
`(eta, lambda2, lambda_d, beta1, beta2) = (0.12, 0.07, 0.03, 0.82, 0.91)`.
NumPy reconstructs the moments, bias corrections, decoupled decay, validation
loss, all five central-difference gradient components, and a directional JVP.
The oracle is checked for deterministic repeated evaluation before timing.

Run:

```bash
python3 scripts/bench_adamw_beta_hypergradient.py \
  --fortml ../fortml --output results/adamw_beta_hypergradient.csv
```

The current FortML checkout has no release app exporting complete arrays for
this objective. The CSV therefore contains seven independent NumPy `pass`
rows and explicit FortML `unavailable` rows. A future
`fortml_bench_adamw_beta_hypergradient` app must export complete value,
directional-JVP, and five-gradient arrays before any FortML timing is retained;
a checksum-only output will not satisfy the lane. CUDA is not inferred from
the CPU objective: the core API currently reports this full hypergradient path
as CPU-only, so no device timing is claimed here.
