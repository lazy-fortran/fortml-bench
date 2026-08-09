# Generic trainer value clipping

This lane compares the generic trainer's per-coordinate gradient clipping against an independent NumPy quadratic oracle. The release app checks the exact SGD update, diagnostic counter, and schema-8 checkpoint round trip.

FortML revision: 565346061a9b10c9ec8878132a6c00549086d6d9
Benchmark revision: 237a7bf09d944c1cf1a5174c34dabefd54d6456f

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | parameter_1 | 0.1 | 0.0 |
| independent_oracle | pass | parameter_2 | 0.9 | 0.0 |
| release_app | pass | parameter_max_abs_error | 0.0 | 0.0 |
| release_app | pass | value_clipped_steps | 1.0 | 0.0 |
| release_app | pass | checkpoint_equal | 1.0 | 0.0 |
| independent_fortran_oracle | pass | test_trainer | 1.0 | 0.0 |
| device_boundary | unavailable | resident_trainer | nan | 0.0 |
