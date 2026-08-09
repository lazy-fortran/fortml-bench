# MLP loss scaling

The lane compares the release app with an independent NumPy recurrence.
The policy starts at 8, grows by 2 after two finite updates, and backs
off by 0.5 after an overflow. The FP64 trainer row checks persisted
dynamic state and explicit scale/check/unscale gradient products. The
FP32 overflow row checks a finite forward/gradient boundary whose
scaled vector overflows: the update is skipped, the scale backs off,
and MLP_EVENT_UPDATE_SKIPPED is delivered exactly once.
FP32 rows compare binary64 master parameters with an independently
rounded NumPy recurrence and check schema-11 checkpoint metadata.
FP16, BF16, and CUDA rows record typed capability boundaries.
Growth and overflow branches are discrete, so smooth HPO products are
not claimed across a branch change.

FortML revision: `fb28717d296030d94416ceea2ec4519a83c93e3f`
Benchmark revision: `bd66ce5ed7984ce2b75576293deb20661141ce8d`

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_recurrence | pass | final_scale | 8.0 | 0.0 |
| independent_recurrence | pass | good_steps | 0.0 | 0.0 |
| independent_recurrence | pass | overflow_count | 1.0 | 0.0 |
| independent_recurrence | pass | skipped_updates | 1.0 | 0.0 |
| release_app_recurrence | pass | final_scale | 8.0 | 1.4901161138336505e-09 |
| fp64_training_state | pass | loss_scale | 16.0 | 0.0 |
| gradient_scale_round_trip | pass | max_abs_error | 0.0 | 0.0 |
| gradient_overflow_detection | pass | overflow_detected | 1.0 | 0.0 |
| gradient_overflow_commit | pass | status_code | 2.0 | 0.0 |
| fp32_overflow_skip | pass | skipped_updates | 1.0 | 0.0 |
| fp32_master_trajectory | pass | max_abs_error | 1.4901161138336505e-09 | 1.4901161138336505e-09 |
| fp32_checkpoint | pass | precision_kind | 2.0 | 0.0 |
| fp16_typed_refusal | pass | status_code | 3.0 | 0.0 |
| bf16_typed_refusal | pass | status_code | 3.0 | 0.0 |
| cuda_typed_refusal | unavailable | resident_loss_scaling | nan | 0.0 |
| independent_fortran_oracle | pass | test_mlp_loss_scaling | 1.0 | 0.0 |
| independent_fortran_event_oracle | pass | test_mlp_training | 1.0 | 0.0 |
