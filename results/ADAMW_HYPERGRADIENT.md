# AdamW training and MLP hypergradient workloads

`scripts/bench_neural_training.py` writes the two raw records
[`adamw_training.csv`](adamw_training.csv) and
[`mlp_hypergradient.csv`](mlp_hypergradient.csv).  The fixture uses the same
deterministic 96-by-3, 3-8-1 tanh MLP as the production trainer lane and
float64 arithmetic.

The AdamW lane runs 24 full-batch updates with learning rate `0.01`,
`beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, decoupled weight decay `0.01`, and
an objective L2 coefficient of `1e-4`.  The NumPy oracle independently
implements the first/second moment recurrences, bias correction, decay
factor, complete prediction vector, and initial/final loss.  Its recorded
final loss is `1.5237110050523792e-3`; this is a behavioral value, not a
cross-machine speed claim.

The hypergradient lane splits the fixture into 72 training and 24 validation
rows.  It performs exactly eight full-batch SGD updates and treats
`[log(learning_rate), log(L2)]` as the outer variables.  The independent
NumPy oracle computes the validation MSE, both log-coordinate gradient
components, and a fixed directional JVP with central differences (`h=2e-5`)
of the complete trajectory.  The recorded values are:

| quantity | value |
|---|---:|
| validation MSE | `2.292893081608181e-2` |
| d/d log learning-rate | `-1.696186500976804e-2` |
| d/d log L2 | `4.617336027412655e-7` |
| direction `[0.7,-0.3]` JVP | `-1.187344402823684e-2` |

Run both lanes serially:

```bash
.venv/bin/python -B scripts/bench_neural_training.py \
  --fortml ../fortml \
  --adamw-output results/adamw_training.csv \
  --hypergradient-output results/mlp_hypergradient.csv
```

The harness builds with `fo build --flag -O3` and then attempts the release
targets `fortml_bench_adamw_training` and
`fortml_bench_mlp_hypergradient`.  A target is accepted only when its oracle
file contains every prediction, scalar, gradient component, and JVP and the
values agree with the independent NumPy record.  A missing target or build
failure is emitted as explicit `unavailable` rows, never silently dropped.
The current record therefore retains the NumPy pass rows and the FortML
target boundary.  It also records untimed CUDA refusal rows for each phase:
the current trainer and hypergradient release apps are host-only, and no CPU
timing is relabeled as CUDA evidence.

The AdamW target protocol uses `FORTML_BENCH_ADAMW_ORACLE` and CSV quantities
`prediction`, `initial_loss`, and `final_loss`; it emits
`mlp_adamw_fit,<...>,seconds` and `mlp_adamw_predict,<...>,seconds` records.
The hypergradient target uses `FORTML_BENCH_HYPERGRADIENT_ORACLE` and
quantities `value`, `gradient` (indices 1 and 2), and `jvp`; it emits
`mlp_hypergradient_value_gradient,<...>,seconds`.  These protocols make the
release-app integration a reproducible follow-on rather than a guessed timing
row.
