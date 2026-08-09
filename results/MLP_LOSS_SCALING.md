# MLP loss scaling

The lane compares the release app with an independent NumPy recurrence.
The policy starts at 8, grows by 2 after two finite updates, and backs
off by 0.5 after an overflow. The FP64 trainer row checks persisted
dynamic state and explicit scale/check/unscale gradient products. FP32
and CUDA rows record typed capability boundaries.
Growth and overflow branches are discrete, so smooth HPO products are
not claimed across a branch change.

FortML revision: `8249266631068de506a58b261cd874b3a9ce2c07`
Benchmark revision: `f36d292418cb4488b837c49ce65a092bf74ec53b+dirty`

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_recurrence | pass | final_scale | 8.0 | 0.0 |
| independent_recurrence | pass | good_steps | 0.0 | 0.0 |
| independent_recurrence | pass | overflow_count | 1.0 | 0.0 |
| independent_recurrence | pass | skipped_updates | 1.0 | 0.0 |
| release_app_recurrence | pass | final_scale | 8.0 | 0.0 |
| fp64_training_state | pass | loss_scale | 16.0 | 0.0 |
| gradient_scale_round_trip | pass | max_abs_error | 0.0 | 0.0 |
| gradient_overflow_detection | pass | overflow_detected | 1.0 | 0.0 |
| gradient_overflow_commit | pass | status_code | 2.0 | 0.0 |
| fp32_typed_refusal | pass | status_code | 3.0 | 0.0 |
| cuda_typed_refusal | unavailable | resident_loss_scaling | nan | 0.0 |
| independent_fortran_oracle | pass | test_mlp_loss_scaling | 1.0 | 0.0 |
