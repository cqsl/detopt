contains the user-facing scripts

- `sld-optimize`: optimize all determinants of the ground state at once; `--project_spin`, `--project_irrep`, `--project_lz` optimize the projected state instead (variation after projection)
- `sld-optimize-partial`: optimize a subset of the determinants per step, the linear coefficients of the rest
- `sld-optimize-partial-distributed`: the same with `jax.distributed` on multiple nodes
- `sld-postprocess`: energy, variance and S^2 of an output file
- `sld-postprocess-symm`: point-group symmetry, Lz and dipole moment of the state; `--project_spin S ...` adds the spin, point-group and Lz projectors
- `sld-show-symm`: assign irreps and print the state and projector weights saved by `sld-postprocess-symm`

run any of them with `--help` for the options
