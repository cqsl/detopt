Energies are in Hartree, dipoles in atomic units, the `xyz` coordinates in Ångström. Strings are stored as bytes.

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
| `eigvals`  | Float64    | (n_steps,) | value of the loss at every step: the lowest eigenvalue of the generalized eigenvalue problem; includes the penalty term (`s2_loss`); with a projector (`project_*`) the projected energy ⟨Ψ\|Ĥ P̂\|Ψ⟩/⟨Ψ\|P̂\|Ψ⟩ |
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
| `project_spin`, `project_irrep`, `project_lz`, `linear_subgroup` | projector of `sld-optimize` (variation after projection); the determinants are those of \|Ψ⟩, the state is P̂\|Ψ⟩ |

## Postprocess Group (`/postprocess`)

| Name | Description |
|------|-------------|
| `backend`, `nonorthogonal`, `skipvar`, `skipdip`, `load_integrals_mo`, `project_spin`, `projected_energy`, `projected_variance` | options of the postprocessing run that wrote the file (only those that apply to the script) |

### Ground state

Written by `sld-postprocess`.

| Name  | Description |
|-------|-------------|
| `e`   | energy ⟨Ĥ⟩ |
| `s2`  | ⟨Ŝ²⟩ |
| `var` | variance ⟨Ĥ²⟩ - ⟨Ĥ⟩² |
| `var_fc` | variance of the frozen-core Hamiltonian, for runs with frozen core orbitals |
| `<op>_mels` | ⟨φ_i\|Ô\|φ_j⟩ for `o`, `e`, `e2`, `s2` (`e2_fc` for runs with frozen core orbitals) |

### Symmetry

Written by `sld-postprocess-symm`, for the state |ψ⟩ = Σ_i |φ_i⟩ with |φ_i⟩ the normalized determinants.

| Name | Description |
|------|-------------|
| `<op>` | ⟨ψ\|Ô\|ψ⟩ / ⟨ψ\|ψ⟩ |
| `<op>_matrix` | ⟨ψ\|Ô\|ψ⟩, shape (1, 1) |
| `<op>_mels` | ⟨φ_i\|Ô\|φ_j⟩ (with `--save_mels`) |

| `<op>` | Ô |
|--------|----------|
| `o` | 1̂ (identity), the keys hold the overlaps |
| `e`, `e2` | Ĥ, Ĥ² |
| `s2` | Ŝ² |
| `lz` | L̂z (linear molecules) |
| `mux`, `muy`, `muz` | dipole μ̂ |
| `E`, `C_2`, `sigma_v(0)`, ... | point-group operation ĝ |
| `PS<S>_<g>` | P̂_S ĝ, with P̂_S the projector onto total spin S |
| `PS<S>_PLz<i>_<g>` | P̂_S ĝ exp(-iθ_i L̂z), θ_i = 2π i / N, i = 0 … N-1 (linear molecules, see below) |
| `hp_<op>`, `h2p_<op>` | ĤÔ and Ĥ²Ô for the two above, for the projected energies and variances |

### Point group and character table

The point group of the molecule. The weight of irrep Γ in the state is ⟨ψ|P̂_Γ|ψ⟩ / ⟨ψ|ψ⟩, with

    P̂_Γ = Σ_g w_Γg ĝ

computed from the `<g>` keys above.

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `symm_group` | String | | point group, e.g. `C2v`, `Coov` (C∞v), `Dooh` (D∞h) |
| `symm_linear` | Bool | | True for a linear molecule |
| `symm_symbols` | String | (n_symbols,) | the operations ĝ, the `<g>` in the keys above, e.g. `E`, `C_2`, `sigma_v(0)`, `sigma_d(0)` for C2v |
| `symm_irreps` | String | (n_irreps,) | the irreps Γ, e.g. `A_1`, `A_2`, `B_1`, `B_2` for C2v |
| `symm_characters` | Float64 | (n_irreps, n_symbols) | the weights w_Γg, for C2v w_Γg = χ_Γ(g) / 4 = ±0.25 |

Linear molecules have no `symm_irreps` and `symm_characters` (e.g. `symm_symbols` = `E`, `sigma_v` for LiH). The projector onto Lz = m is

    P̂_m = (1/N) Σ_i exp(imθ_i) exp(-iθ_i L̂z)

so the weights of Lz = m are the discrete Fourier transform over i of the `PS<S>_PLz<i>_<g>` values.
