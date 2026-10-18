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
| `eigvals`  | Float64    | (n_steps,) | value of the loss at every step: the lowest eigenvalue of the generalized eigenvalue problem for the ground state, tr(S⁻¹H) over the states for excited states; includes the penalty terms (`s2_loss`, `symm_loss`); with a projector (`project_*`) the projected energy ⟨Ψ\|Ĥ P̂\|Ψ⟩/⟨Ψ\|P̂\|Ψ⟩ |
| `mo_coeff` | Float64    | (n_orb, n_orb) | the orthonormal basis (not necessarily the Hartree-Fock orbitals) |

The determinants are not normalized. For excited states the determinants are ordered by state, `nper` per state.

## Extra Group (`/extra`)

Settings of the optimization.

| Name | Description |
|------|-------------|
| `seed` | random seed |
| `charge` | total charge of the molecule (`--charge`), only for ions |
| `ghf`, `real`, `nonorthogonal`, `dense_solver`, `particle_hole`, `orthogao`, `start`, `s2_loss` | options of the optimization |
| `uhf_backend`, `hybrid_*`, `lanczos_*`, `static_spin_blocks`, `batch_size_hess` | options of the matrix element backend and the solver |
| `nstates`, `nper`, `sweep`, `symm_loss`, `uniform_spin_count` | options of `sld-optimize-exc` (number of states, determinants per state, ...) |
| `energy_vals` | energies of the states at every step, (n_steps, nstates), only `sld-optimize-exc` |
| `partialopt` | option of `sld-optimize-partial`, number of determinants whose orbitals are optimized per step |
| `project_spin`, `project_irrep`, `project_lz`, `linear_subgroup` | projector of `sld-optimize` (variation after projection); the determinants are those of \|Ψ⟩, the state is P̂\|Ψ⟩ |

## Postprocess Group (`/postprocess`)

| Name | Description |
|------|-------------|
| `backend`, `nonorthogonal`, `skipvar`, `skipdip`, `load_integrals_mo`, `project_spin`, `projected_energy`, `projected_variance`, `trdm` | options of the postprocessing run that wrote the file (only those that apply to the script) |

### Ground state

Written by `sld-postprocess`.

| Name  | Description |
|-------|-------------|
| `e`   | energy ⟨Ĥ⟩ |
| `s2`  | ⟨Ŝ²⟩ |
| `var` | variance ⟨Ĥ²⟩ - ⟨Ĥ⟩² |
| `var_fc` | variance of the frozen-core Hamiltonian, for runs with frozen core orbitals |
| `<op>_mels` | ⟨φ_i\|Ô\|φ_j⟩ for `o`, `e`, `e2`, `s2` (`e2_fc` for runs with frozen core orbitals) |

### Excited states

Written by `sld-postprocess-symm` and `sld-diagonalize-symm`. K is the size of the subspace, i.e. the number of states.
The states are

    |ψ_a⟩ = Σ_i c_ai |φ_i⟩,   a = 0 … K-1

with |φ_i⟩ the normalized determinants. For `_all` they are expanded in the optimized states |χ_b⟩ instead,

    |ψ_a⟩ = Σ_b c_ab |χ_b⟩,   |χ_b⟩ = Σ_{i ∈ b} x_i |φ_i⟩

with i ∈ b the `nper` determinants of state b and x_i the norm of the stored determinant i.

| Name | Shape | Description |
|------|-------|-------------|
| `<op><suffix>` | (K,) | ⟨ψ_a\|Ô\|ψ_a⟩ / ⟨ψ_a\|ψ_a⟩ |
| `<op>_matrix<suffix>` | (K, K) | ⟨ψ_a\|Ô\|ψ_b⟩ |
| `<op>_mels` | (n_determinants, n_determinants) | ⟨φ_i\|Ô\|φ_j⟩ |
| `c<suffix>` | (K, n_determinants) | c_ai; for `_all` c_ab, (K, K) |
| `tmux<suffix>`, `tmuy<suffix>`, `tmuz<suffix>` | (K, K) | transition dipole moments ⟨ψ_a\|μ̂\|ψ_b⟩ between the normalized states |
| `trdm<suffix>` | (2, K, K, n_orb, n_orb) | transition 1-RDMs, `trdm[s, a, b, p, q]` = ⟨ψ_a\|ĉ†_ps ĉ_qs\|ψ_b⟩ between the normalized states, in the `mo_coeff` basis, s = up, down; a = b is the 1-RDM of each state |
| `rotation_full_degenerate_diag` | (K, K) | unitary U from the `_full` states ψ to the `_full_degenerate_diag` states ψ', \|ψ'_a⟩ = Σ_b U_ab \|ψ_b⟩, so `c_full_degenerate_diag` = U `c_full`; it only mixes (near) degenerate states, so that Ŝ², L̂z or the point group are diagonal among them |

| `<op>` | Ô |
|--------|----------|
| `o` | 1̂ (identity), the keys hold the overlaps |
| `e`, `e2` | Ĥ, Ĥ² |
| `e2_fc` | Ĥ_fc², with Ĥ_fc the frozen-core Hamiltonian; replaces `e2` for runs with frozen core orbitals |
| `s2` | Ŝ² |
| `sz` | Ŝz |
| `lz` | L̂z (linear molecules) |
| `mux`, `muy`, `muz` | dipole μ̂ |
| `E`, `C_2`, `sigma_v(0)`, ... | point-group operation ĝ |
| `PS<S>_<g>` | P̂_S ĝ, with P̂_S the projector onto total spin S |
| `PS<S>_PLz<i>_<g>` | P̂_S ĝ exp(-iθ_i L̂z), θ_i = 2π i / N, i = 0 … N-1 (linear molecules, see below) |
| `hp_<op>`, `h2p_<op>` | ĤÔ and Ĥ²Ô for the two above, for the projected energies and variances |

| `<suffix>` | States |
|------------|--------|
| (none) | single state |
| `_all` | the optimized states rediagonalized among themselves, Σ_b H_ab v_b = E Σ_b S_ab v_b in the span of the K optimized states |
| `_full` | the K lowest solutions of Σ_j H_ij c_j = E Σ_j S_ij c_j, with H_ij = ⟨φ_i\|Ĥ\|φ_j⟩ and S_ij = ⟨φ_i\|φ_j⟩ over all determinants |
| `_full_degenerate_diag` | `_full` rotated by `sld-diagonalize-symm`, \|ψ'_a⟩ = Σ_b U_ab \|ψ_b⟩ |

### Point group and character table

The point group of the molecule, the same for all states. The weight of irrep Γ in state a is ⟨ψ_a|P̂_Γ|ψ_a⟩ / ⟨ψ_a|ψ_a⟩, with

    P̂_Γ = Σ_g w_Γg ĝ

computed from the `<g><suffix>` keys above.

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
