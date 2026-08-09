# Trainer partial-fit contract

This lane checks the generic trainer_t%partial_fit warm-start contract against an independent NumPy Adam recurrence. One six-update trajectory is compared with two and four update chunks, then the Fortran test checks checkpoint continuation and transactional over-budget requests. CUDA is an explicit typed refusal because the generic trainer has no resident objective or optimizer state.

FortML revision: 7448208a60c8295b120dbd8f7c53368226ba2861
Benchmark revision: 6a4482ec836d01473b3e690558be4f4e73218506

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | replay_max_abs_error | 0.0 | 0.0 |
| independent_oracle | pass | final_parameter_norm | 1.0364655023448386 | 0.0 |
| release_app | pass | replay_max_abs_error | 0.0 | 0.0 |
| release_app | pass | steps | 6.0 | 0.0 |
| cuda_typed_refusal | unavailable | partial_fit_status_code | 3.0 | 0.0 |
| independent_fortran_oracle | pass | test_trainer_partial_fit | 1.0 | 0.0 |
