# Generic trainer learning-rate schedules

This lane uses an independent one-cycle formula and quadratic SGD
recurrence. The release app is accepted only when all four schedule
rates and both final parameters agree within the stated tolerance.

FortML revision: d767e9cbfdb07680aa17d544009809293603a0ed
Benchmark revision: 01aa57a84565e167990b744d99381e9850184317

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | rate_1 | 0.15 | 0.0 |
| independent_oracle | pass | rate_2 | 0.2 | 0.0 |
| independent_oracle | pass | rate_3 | 0.1125 | 0.0 |
| independent_oracle | pass | rate_4 | 0.025 | 0.0 |
| release_schedule | pass | rate_max_abs_error | 2.7755575615628914e-17 | 2.7755575615628914e-17 |
| release_recurrence | pass | parameter_max_abs_error | 5.551115123125783e-17 | 5.551115123125783e-17 |
| cuda_typed_refusal | unavailable | resident_schedule_optimizer |  | 0.0 |
