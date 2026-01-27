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


def A_inv_det_stable_2(A):
    # section 32.4
    # this is for the difference of two MM-MM (instead of Ht)
    X, Σ, YH = jnp.linalg.svd(A)
    n = A.shape[0]

    I = jnp.eye(n, dtype=bool)
    I = I[:, :, None] | I[:, None, :]

    s = (I + (~I) * Σ[:, None, None]).prod(axis=0)
    # replacement for det A
    D = jnp.linalg.det(X) * jnp.linalg.det(YH)
    # replacement for A_inv
    I = jnp.eye(n, dtype=bool)
    MM = jnp.einsum("bc,jb,lc,bi,ck->ijkl", (~I) * s, X, X, YH, YH, optimize="optimal") - jnp.einsum("bc,lb,jc,bi,ck->ijkl", (~I) * s, X, X, YH, YH, optimize="optimal")
    MM = MM.conj()  # conj X and Y* - > X* Y
    return D, MM


def A_inv_det_stable_3(A):
    # section 32.2.3

    X, Σ, YH = jnp.linalg.svd(A)
    n = A.shape[0]

    I = jnp.eye(n, dtype=bool)
    # I2 = I[:, :, None ] | I[:, None, :] |  I[None, :, :]
    I2 = jnp.array(False)
    I3 = I[:, :, None, None] | I[:, None, :, None] | I[:, None, None, :]

    s = (I3 + (~I3) * Σ[:, None, None, None]).prod(axis=0)
    DD = jnp.linalg.det(X) * jnp.linalg.det(YH)

    MMM = 0
    MMM += jnp.einsum("bcd,bj,ib,cl,kc,dn,md->ijklmn", (~I2) * s, YH, X, YH, X, YH, X, optimize="optimal")
    MMM += jnp.einsum("bcd,cj,ib,dl,kc,bn,md->ijklmn", (~I2) * s, YH, X, YH, X, YH, X, optimize="optimal")
    MMM += jnp.einsum("bcd,dj,ib,bl,kc,cn,md->ijklmn", (~I2) * s, YH, X, YH, X, YH, X, optimize="optimal")
    MMM -= jnp.einsum("bcd,bj,ib,dl,kc,cn,md->ijklmn", (~I2) * s, YH, X, YH, X, YH, X, optimize="optimal")
    MMM -= jnp.einsum("bcd,dj,ib,cl,kc,bn,md->ijklmn", (~I2) * s, YH, X, YH, X, YH, X, optimize="optimal")
    MMM -= jnp.einsum("bcd,cj,ib,bl,kc,dn,md->ijklmn", (~I2) * s, YH, X, YH, X, YH, X, optimize="optimal")
    MMM = MMM.conj()  # conj X and Y* - > X* Y; TODO check
    return DD, MMM


def T(Aconj, B, m):
    A = Aconj @ B.T
    X, Σ, YH = jnp.linalg.svd(A)
    n = A.shape[0]
    I = jnp.eye(n, dtype=bool)
    c = [
        slice(None),
    ] + [
        None,
    ] * m
    Im = False
    for i in range(m):
        c_ = c.copy()
        c_[i + 1] = slice(None)
        Im = Im | I[*c_]
    s = (Im + (~Im) * Σ[*c]).prod(axis=0)
    D = jnp.linalg.det(X) * jnp.linalg.det(YH)
    return D, s, B.T @ YH.T.conj(), X.T.conj() @ Aconj
