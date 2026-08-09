# MLP checkpoint fingerprint

This lane checks deterministic checkpoint identity with an independent decimal-token oracle and the FortML behavioral test. The release test covers formatted round-trip equality, optimizer-state and metadata mutation detection, invalid-state zero, and the CPU/CUDA boundary. CUDA is unavailable until a resident trainer exposes an explicit device-to-host snapshot.

FortML revision: 94378bbdc43183c5813467699847b6b3d171af2f
Benchmark revision: eb7df114df178b9ef8617852057d166de317201d

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | round_trip_equal | 1.0 | 0.0 |
| independent_oracle | pass | state_mutation_detected | 1.0 | 0.0 |
| independent_oracle | pass | metadata_mutation_detected | 1.0 | 0.0 |
| independent_oracle | pass | invalid_fingerprint_zero | 1.0 | 0.0 |
| public_contract_gate | pass | test_mlp_checkpoint_fingerprint | 1.0 | 0.0 |
| cuda_typed_refusal | unavailable | resident_checkpoint_fingerprint | nan | 0.0 |
