# MLP loss scaling

The lane compares the release app with an independent NumPy recurrence.
The policy starts at 8, grows by 2 after two finite updates, and backs
off by 0.5 after an overflow. The FP64 trainer row checks persisted
dynamic state. FP32 and CUDA rows record typed capability boundaries.

FortML revision: `f1c8ac26c1510ce07dde5a8fd85c2c6aa12a0bd7`  
Benchmark revision: `7343f613cc2a64ecab6d650d6a6f4403922e55af+dirty`

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_recurrence | pass | final_scale | 8.0 | 0.0 |
| independent_recurrence | pass | good_steps | 0.0 | 0.0 |
| independent_recurrence | pass | overflow_count | 1.0 | 0.0 |
| independent_recurrence | pass | skipped_updates | 1.0 | 0.0 |
| release_app_recurrence | pass | final_scale | 8.0 | 0.0 |
| fp64_training_state | pass | loss_scale | 16.0 | 0.0 |
| fp32_typed_refusal | pass | status_code | 3.0 | 0.0 |
| cuda_typed_refusal | unavailable | resident_loss_scaling | nan | 0.0 |
| independent_fortran_oracle | pass | test_mlp_loss_scaling | 1.0 | 0.0 |
