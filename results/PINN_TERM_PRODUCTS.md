# Named PINN term products

This lane checks the per-term diagnostics introduced for the shared
\`physics_objective_t\`/\`pinn_training_adapter_t\` seam. The independent NumPy
row evaluates a weighted quadratic residual
\`2 (theta^2 - 0.25)^2 / 2\`, including its exact parameter gradient and HVP.
The FortML release row runs \`fortml_bench_pinn_term_products\` and checks the
residual column and the three inactive named-term zero columns. The CUDA row
is a typed unavailable boundary: no resident PINN residual graph or host
fallback is claimed.

\`\`\`bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_pinn_term_products.py \\
  --fortml ../fortml --output results/pinn_term_products.csv
\`\`\`

The CSV stores source and benchmark revisions, the independent oracle,
release-app timing, and the explicit CUDA capability row.
