Energies are in Hartree, the `xyz` coordinates in Ångström. Strings are stored as bytes.

## Root Group (`/`)

### Attributes

| Name      | Type    | Description |
|-----------|---------|-------------|
| `basis`   | String  | basis set name (e.g. `ccpvdz`) |
| `name`    | String  | name of the input file |
| `spin`    | Integer | "spin" as in pyscf, equivalent to 2 * Sz |
| `xyz`     | String  | xyz coordinates of the molecule |
| `cart`    | Bool    | cartesian instead of spherical basis functions |
| `nfreeze` | Integer | number of frozen-core electrons (only with `--nfreeze`) |

### Datasets

| Name       | Type       | Shape | Description |
|------------|------------|-------|-------------|
| `Xsu`      | Complex128 | (n_determinants, n_electron_up, n_orb) | UHF determinants in the orthonormal basis, spin-up part |
| `Xsd`      | Complex128 | (n_determinants, n_electron_down, n_orb) | UHF determinants in the orthonormal basis, spin-down part |
| `Xs`       | Complex128 | (n_determinants, n_electron, 2 n_orb) | GHF determinants, replaces `Xsu` and `Xsd` |
| `eigvals`  | Float64    | (n_steps,) | value of the loss at every step: the lowest eigenvalue of the generalized eigenvalue problem; includes the penalty term (`s2_loss`) |
| `mo_coeff` | Float64    | (n_orb, n_orb) | the orthonormal basis (not necessarily the Hartree-Fock orbitals) |

The determinants are not normalized.

## Extra Group (`/extra`)

Settings of the optimization.

| Name | Description |
|------|-------------|
| `seed` | random seed |
| `charge` | total charge of the molecule (`--charge`), only for ions |
| `ghf`, `real`, `nonorthogonal`, `dense_solver`, `particle_hole`, `orthogao`, `start`, `s2_loss` | options of the optimization |
| `uhf_backend`, `hybrid_*`, `lanczos_*`, `static_spin_blocks`, `batch_size_hess` | options of the matrix element backend and the solver |
| `partialopt` | option of `sld-optimize-partial`, number of determinants whose orbitals are optimized per step |

## Postprocess Group (`/postprocess`)

| Name | Description |
|------|-------------|
| `backend`, `nonorthogonal`, `skipvar`, `load_integrals_mo` | options of the postprocessing run that wrote the file |

### Ground state

Written by `sld-postprocess`.

| Name  | Description |
|-------|-------------|
| `e`   | energy ⟨Ĥ⟩ |
| `s2`  | ⟨Ŝ²⟩ |
| `var` | variance ⟨Ĥ²⟩ - ⟨Ĥ⟩² |
| `var_fc` | variance of the frozen-core Hamiltonian, for runs with frozen core orbitals |
| `<op>_mels` | ⟨φ_i\|Ô\|φ_j⟩ for `o`, `e`, `e2`, `s2` (`e2_fc` for runs with frozen core orbitals) |
