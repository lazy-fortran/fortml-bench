# Structured tensor-product workload

This record transfers the independently tested `fortnum_tensor_product`
OpenACC workload into the cross-engine evidence repository. It is a regular
grid contraction rather than a pairwise kernel sum, so KeOps and GPyTorch-KeOps
are not direct competitors for this mathematical operator. The matched
reference is the ordinary nvfortran host contraction with the same float64
factors and four right-hand sides.

`enter_data(status, n_rhs)` copies the factors and allocates persistent
contraction workspaces. The transfer rows open an input/output data region for
each operation. The resident rows keep the input, output, factors, and work
arrays in one data region across all repetitions. The 512-sample case passes an
independent explicit dense Kronecker oracle; the 4096-sample case uses the
independent tensor contraction oracle without materializing the dense matrix.

| N | host (ms) | device + transfer (ms) | device resident (ms) | host/resident |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.0369 | 0.0811 | 0.0652 | 0.57x |
| 4096 | 0.8427 | 0.1779 | 0.1261 | 6.68x |

The source implementation and low-level report are in fortnum commit
`73e8965`. The inspected scaling plot is
https://box.sloppy.at/4c393.png. This workload confirms the OpenACC structured
path; compact-support sparse dispatch through `fortsparse` remains a separate
benchmark gate.
