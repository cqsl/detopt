import jax.numpy as jnp


def A_inv_det_stable_1(A):
    # section 32.3
    X, Σ, YH = jnp.linalg.svd(A)
    n = A.shape[0]
    I = jnp.eye(n, dtype=bool)
    s = (I + (~I) * Σ[:, None]).prod(axis=0)
    # replacement for det A
    D = jnp.linalg.det(X) * jnp.linalg.det(YH)
    # replacement for A_inv
    M = jnp.einsum("b,jb,bi->ij", s, X, YH, optimize="optimal").conj()
    return D, M


def T2(Σ, m):
    n = Σ.shape[0]
    I = jnp.eye(n, dtype=bool)
    c = [slice(None),] + [None,] * m
    Im = False
    for i in range(m):
        c_ = c.copy()
        c_[i + 1] = slice(None)
        Im = Im | I[*c_]
    s = (Im + (~Im) * Σ[*c]).prod(axis=0)
    return s


def T(Aconj, B, m, return_svd=False):
    A = Aconj @ B.T
    X, Σ, YH = jnp.linalg.svd(A)
    s = T2(Σ, m)
    D = jnp.linalg.det(X) * jnp.linalg.det(YH)
    U = B.T @ YH.T.conj()
    V = X.T.conj() @ Aconj
    if return_svd:
        return D, s, U, V, X, YH
    return D, s, U, V
