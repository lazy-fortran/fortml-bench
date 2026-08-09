# MLP optimizer-group registry benchmark

This lane checks that named optimizer groups survive formatted checkpoint round-trips and that resume rejects identity drift even when ranges and multipliers are unchanged. The CUDA row records the typed refusal for the non-resident grouped hypergradient path.

FortML revision: c50511c45eb142649f13f2a9bc12f867a1f1100a
Benchmark revision: 4a931bae2c8fa4bd2c134704c0f50687f22b8ef2

| phase | device | status | metric | value |
| --- | --- | --- | --- | --- |
| independent_oracle | cpu | pass | registry_name | bias |
| release_app | cpu | pass | registry_name | bias |
| behavioral_gate | cpu | pass | name_drift_status | 1 |
| device_boundary | cuda | unavailable | optimizer_group_status | 3 |
