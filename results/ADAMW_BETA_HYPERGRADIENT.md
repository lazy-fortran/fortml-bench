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
loss, all five central-difference gradient components, a directional JVP, and
the five-component outer HVP obtained by differentiating that independent
gradient oracle along the same direction.
The oracle is checked for deterministic repeated evaluation before timing.

Run:

```bash
python3 scripts/bench_adamw_beta_hypergradient.py \
  --fortml ../fortml --output results/adamw_beta_hypergradient.csv
```

The release app exports complete value, directional-JVP, five-gradient, and
five-HVP arrays before its timing is retained. The CPU HVP is analytic on the
one-layer linear fixture. Nonlinear and multilayer requests return a typed
third-derivative refusal, and CUDA is not inferred from the CPU objective: the
full hypergradient path remains CPU-only until resident state derivatives are
available.
