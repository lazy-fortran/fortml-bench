# MLP checkpoint fingerprint

This lane checks deterministic checkpoint identity with an independent decimal-token oracle and the FortML behavioral test. The release test covers formatted round-trip equality, optimizer-state and metadata mutation detection, invalid-state zero, and the CPU/CUDA boundary. CUDA is unavailable until a resident trainer exposes an explicit device-to-host snapshot.

FortML revision: 2c7b503803c1db02d3dba625f85f1fcaac96a51b
Benchmark revision: e4da1310daadf5006d744d7fc867616e9860824c

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | round_trip_equal | 1.0 | 0.0 |
| independent_oracle | pass | state_mutation_detected | 1.0 | 0.0 |
| independent_oracle | pass | metadata_mutation_detected | 1.0 | 0.0 |
| independent_oracle | pass | invalid_fingerprint_zero | 1.0 | 0.0 |
| public_contract_gate | pass | test_mlp_checkpoint_fingerprint | 1.0 | 0.0 |
| cuda_typed_refusal | unavailable | resident_checkpoint_fingerprint | nan | 0.0 |
