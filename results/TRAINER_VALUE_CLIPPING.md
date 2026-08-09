# Generic trainer value clipping

This lane compares the generic trainer's per-coordinate gradient clipping against an independent NumPy quadratic oracle. The release app checks the exact SGD update, diagnostic counter, and schema-8 checkpoint round trip.

FortML revision: 188c222655d4e5b10a5f3c5dec725ab39a41d72e
Benchmark revision: 844dfd1c6d314adb64a46ad4c80ea24cbbdaec73

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | parameter_1 | 0.1 | 0.0 |
| independent_oracle | pass | parameter_2 | 0.9 | 0.0 |
| release_app | pass | parameter_max_abs_error | 0.0 | 0.0 |
| release_app | pass | value_clipped_steps | 1.0 | 0.0 |
| release_app | pass | checkpoint_equal | 1.0 | 0.0 |
| independent_fortran_oracle | pass | test_trainer | 1.0 | 0.0 |
| device_boundary | unavailable | resident_trainer | nan | 0.0 |
