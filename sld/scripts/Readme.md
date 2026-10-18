contains the user-facing scripts

- `sld-optimize`: optimize all determinants of the ground state at once
- `sld-optimize-partial`: optimize a subset of the determinants per step, the linear coefficients of the rest
- `sld-optimize-partial-distributed`: the same with `jax.distributed` on multiple nodes
- `sld-postprocess`: energy, variance and S^2 of an output file

run any of them with `--help` for the options
