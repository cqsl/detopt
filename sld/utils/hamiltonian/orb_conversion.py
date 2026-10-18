import jax.numpy as jnp

import numpy as np

def spinorb_one_body(hij):
    # Σ_ijσ h_ij c†_iσ c_jσ, spin orbitals ordered n*spin+i
    return np.kron(np.eye(2), np.asarray(hij))


def spin_squared_two_body(n):
    # S^2 = 3/4 N + Σ_a Σ s^a_pq s^a_rs c†_p c†_r c_s c_q, with s^a = σ_a/2 on the spin of the spin orbitals
    sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]) / 2
    s = np.stack([np.kron(x, np.eye(n)) for x in sigma])
    return np.einsum("apq,ars->prsq", s, s).real


def spinorb_from_spatial2(hij, hijkl):
    # Σ_ijklμσ h_ijkl c†_iμ c†_jσ c_kμ c_lσ
    hijkl = np.asarray(hijkl)
    n = hijkl.shape[0]
    u, d = slice(0, n), slice(n, None)
    h4 = np.zeros((2 * n,) * 4, dtype=hijkl.dtype)
    h4[u, u, u, u] = hijkl
    h4[d, d, d, d] = hijkl
    h4[u, d, d, u] = hijkl
    h4[d, u, u, d] = hijkl
    return spinorb_one_body(hij), h4


def uhf2ghf(Xu, Xd):
    return jnp.block([[Xu, jnp.zeros_like(Xu)], [jnp.zeros_like(Xd), Xd]])
