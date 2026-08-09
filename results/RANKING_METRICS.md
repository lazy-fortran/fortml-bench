# Grouped ranking metric benchmark

This lane compares FortML grouped NDCG against an independent Python DCG reduction. It also runs the Fortran behavioral oracle for cutoff, weights, tie order, undefined zero-ideal handling, and the typed CUDA boundary.

FortML revision: dfa3a923c2f2f5ae987462d46550b7d761d2795b
Benchmark revision: bb24135893c4af99eec4d9fbc00dd55d3680aaeb

| phase | device | status | metric | value | max abs error |
| --- | --- | --- | --- | ---: | ---: |
| independent_oracle | cpu | pass | ndcg | 0.8295009024012067 | 0.0 |
| release_app | cpu | pass | ndcg | 0.8295009024012067 | 0.0 |
| behavioral_gate | cpu | pass | test_ranking_metrics | 1.0 | 0.0 |
| device_boundary | cuda | unavailable | ranking_ndcg_status | 3.0 | 0.0 |
