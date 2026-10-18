contains the user-facing scripts

- `sld-optimize`: optimize all determinants of the ground state at once; `--project_spin`, `--project_irrep`, `--project_lz` optimize the projected state instead (variation after projection)
- `sld-optimize-partial`: optimize a subset of the determinants per step, the linear coefficients of the rest
- `sld-optimize-exc`: optimize several states at once, for excited states
- `sld-optimize-partial-distributed`, `sld-optimize-exc-distributed`: the same with `jax.distributed` on multiple nodes
- `sld-postprocess`: energy, variance and S^2 of an output file
- `sld-postprocess-symm`: point-group symmetry, Lz and dipole moments of the states, saved as small matrices in the basis of the states; `--project_spin S ...` adds the spin, point-group and Lz projectors
- `sld-diagonalize-symm`: diagonalize S^2, Lz or the point group among near-degenerate excited states, in an interactive shell (or `--auto`), and save the rotated states
- `sld-show-symm`: assign irreps and print the states and projector weights saved by `sld-postprocess-symm` (the rotated ones if present)

run any of them with `--help` for the options
