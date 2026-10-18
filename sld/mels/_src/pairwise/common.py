import jax.numpy as jnp

from sld.utils.hamiltonian.two_body import AntisymmetricTwoBody


def ol(Uconj, V):
    A = Uconj @ V.T
    return jnp.linalg.det(A)


def antisymmetrized(H2e):
    if isinstance(H2e, AntisymmetricTwoBody):
        Ha = H2e.Ha
        return Ha, 2 * Ha, 4 * Ha
    Ht = H2e - H2e.transpose(0, 1, 3, 2)
    return H2e, Ht, Ht - Ht.transpose(1, 0, 2, 3)


# (1 - r1) v, v (1 - r1) and (1 - r1) M (1 - r1) without building 1 - r1
def r1h_v(L, Ac, v):
    return v - L @ (Ac @ v)


def v_r1h(v, L, Ac):
    return v - (v @ L) @ Ac


def r1h_M_r1h(M, L1, Ac1, L2, Ac2):
    M = M - (M @ L2) @ Ac2
    return M - L1 @ (Ac1 @ M)


def hamiltonian_terms(hamiltonian):
    if hamiltonian in [None, {}]:
        hamiltonian = {'Enuc': 1}
    return tuple(hamiltonian.get(k, 0 if k == 'Enuc' else None) for k in ('H1e', 'H2e', 'Enuc', 'w', 'U', 'U_onsite'))


def red(f, Aconj, k, B, l, *args, grad=False, **kwargs):
    if grad:
        # keep dummy trailing axis
        v = B.at[(l,), :].get().T
        args = v, *args
    Aconj = jnp.delete(Aconj, k, axis=0, assume_unique_indices=True)
    B = jnp.delete(B, l, axis=0, assume_unique_indices=True)
    res = f(Aconj, B, *args, **kwargs)
    assert jnp.issubdtype(k.dtype, jnp.signedinteger)
    sign = 1 - 2 * ((k + l) % 2)  # (-1)**(k+l)
    if isinstance(res, tuple):
        res, *rest = res
        return sign * res, *rest
    else:
        return sign * res
