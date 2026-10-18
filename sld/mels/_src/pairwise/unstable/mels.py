import jax
import jax.numpy as jnp
from functools import partial


from ..common import ol
from sld.utils.sparse import einsum_maybe_sparse


def inv(A):
    # workaround for ShardingTypeError: All inputs to triangular_solve must have the same batch sharding
    # TODO fix it properly
    # I = 0.*A+jnp.eye(A.shape[0], dtype=A.dtype, device=jax.typeof(A).sharding)
    # I = jax.lax.pvary(I, tuple(jax.typeof(A).vma))
    # return jnp.linalg.solve(A, I)
    u, s, vt = jnp.linalg.svd(A)
    return jnp.einsum("ij,j,jk->ki", u.conj(), 1.0 / s, vt.conj())


def rdm1_red(Aconj, B, v=None, return_mat=False):
    # without det M
    M = Aconj @ B.T
    M_inv = inv(M)
    if v is None:
        res = jnp.einsum("ki,kl,lj->ij", B, M_inv, Aconj)
    else:
        res = jnp.einsum("ki,kl,lj,j...->i...", B, M_inv, Aconj, v)
    if return_mat:
        # use svd here and not eig, since batched jax.lax.eig runs on cpu per default on small matrices,
        # and even if you set implementation='CUSOLVER' it is very slow
        _, ev, _ = jax.lax.linalg.svd(M)
        return res, (jnp.abs(ev).min() / jnp.abs(ev).max())
    else:
        return res


def rdm1_full(Uconj, V, v=None, return_mat=False):
    if return_mat:
        res, M = rdm1_red(Uconj, V, v=v, return_mat=True)
        return res * ol(Uconj, V), M
    else:
        return rdm1_red(Uconj, V, v=v) * ol(Uconj, V)


def E1e(Aconj, B, H1e, mo_coeff=None, _red=False):
    if H1e is None:
        return 0.0
    M = Aconj @ B.T
    r1 = rdm1_red(Aconj, B)
    if mo_coeff is None:
        E1_red = jnp.einsum("ba,ab", r1, H1e)
    else:
        r1ao = jnp.einsum("ba,Aa,Bb->AB", r1, mo_coeff, mo_coeff)
        E1_red = jnp.einsum("AB,AB", H1e, r1ao)
    if _red:
        return E1_red
    else:
        return jnp.linalg.det(M) * E1_red


E1e_red = partial(E1e, _red=True)


def E2e(Aconj, B, H2e, mo_coeff=None, _red=False):
    if H2e is None:
        return 0.0
    M = Aconj @ B.T
    r1 = rdm1_red(Aconj, B)

    if mo_coeff is None:
        r1ao = r1
    else:
        r1ao = jnp.einsum("ab,Aa,Bb->AB", r1, mo_coeff, mo_coeff)

    if isinstance(H2e, tuple):
        _, Ht, _ = H2e
        E2_red = jnp.einsum("abge,ea,gb", Ht, r1ao, r1ao)
    else:
        E2_red_a = +einsum_maybe_sparse("abge,ea,gb", H2e, r1ao, r1ao, notation="physicist")
        E2_red_b = -einsum_maybe_sparse("abge,ga,eb", H2e, r1ao, r1ao, notation="physicist")
        E2_red = E2_red_a + E2_red_b
    if _red:
        return E2_red
    else:
        return jnp.linalg.det(M) * E2_red


E2e_red = partial(E2e, _red=True)


def expval(X1c, X2, deriv_indices=None, hamiltonian=None, mo_coeff=None):
    # compute <>, <H>
    # GHF version;
    # non-svd version; assumes overlap is nonzero

    if mo_coeff is not None:
        raise NotImplementedError

    ol_ = ol(X1c, X2)

    if hamiltonian is None:
        H2e, H1e, Enuc = None, None, 1
    else:
        H2e, H1e, Enuc = hamiltonian

    e0_ = Enuc
    e1_ = E1e_red(X1c, X2, H1e, mo_coeff=mo_coeff)
    e2_ = E2e_red(X1c, X2, H2e, mo_coeff=mo_coeff)
    expval = ol_ * (e0_ + e1_ + e2_)
    return ol_[None, None], expval[None, None]


def expval_sq(X1c, X2, deriv_indices=None, hamiltonian=None, mo_coeff=None, _sq=True):
    if mo_coeff is not None:
        raise NotImplementedError  # TODO

    # compute <>, <H^2> and <H>
    # GHF version;
    # svd version; works with zero overlap

    ol_ = ol(X1c, X2)

    if hamiltonian is None:
        return ol_[None, None], ol_[None, None], ol_[None, None]
    else:
        # TODO
        raise NotImplementedError
