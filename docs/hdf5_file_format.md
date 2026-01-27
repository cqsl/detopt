## Root Group (`/`)

### Attributes

| Name   | Type        | Description |
|--------|------------|-------------|
| `basis` | String     | Basis set name (e.g., `ccpvdz`) |
| `name`  | String     | Molecule name |
| `spin`  | Integer    | "spin" as in pyscf, equivalent to 2 * Sz |
| `xyz`   | String     | XYZ coordinates of the molecule|

### Datasets

| Name      | Type        | Shape      | Description |
|-----------|------------|-----------|-------------|
| `Xsd`     | Complex128  | (n_determinants, n_electron_down, n_orb) | coefficients of the UHF determinants in the orthonormal basis, spin-down part |
| `Xsu`     | Complex128  | (n_determinants, n_electron_down, n_orb) | coefficients of the UHF determinants in the orthonormal basis, spin-down part |
| `eigvals` | Float64    | (n_steps,)   | Eigenvalues of the generalized eigenvalue problem during the optimization. Equivalent to the energy if no spin penalty term is used. |
| `mo_coeff`| Float64    | (n_orbitals, n_orbitals)  | coefficient matrix of the orthonormal basis (not necessarily the molecular orbital basis obtained from a hartree-fock calculation) |


#### Notes

- The determinants are not normalized; If you require normalized determinants with separate CI coefficients you need to normalize them explicitly.  
- We use the same orthogonal basis for the spin-up and spin-down parts.
- When using GHF-type determinants `Xsd` and `Xsu` are replaced ba a single `Xs` entry.
---

## Postprocess Group (`/postprocess`)

| Name | Type       | Description |
|------|-----------|-------------|
| `e`   | Complex128 | Post-processed energy ⟨H⟩ |
| `s2`  | Complex128 | Spin-squared expectation value ⟨S²⟩ |
| `var` | Complex128 | Variance of the Hamiltonian, given by ⟨H²⟩-⟨H⟩² |

---
