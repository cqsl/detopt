import jax
import jax.numpy as jnp
from functools import partial


from ..common import antisymmetrized, ol


def inv_and_det(A):
    n = A.shape[-1]
    lu, piv, perm = jax.lax.linalg.lu(A)
    P = jax.nn.one_hot(perm, n, dtype=A.dtype)
    y = jax.lax.linalg.triangular_solve(lu, P, left_side=True, lower=True, unit_diagonal=True)
    Ainv = jax.lax.linalg.triangular_solve(lu, y, left_side=True, lower=False)
    sign = 1 - 2 * (jnp.count_nonzero(piv != jnp.arange(n, dtype=piv.dtype)) % 2)
    # not jnp.diagonal, its gather breaks with the "i" sharding
    return Ainv, sign * (lu * jnp.eye(n, dtype=A.dtype)).sum(axis=-1).prod()


def inv(A):
    return inv_and_det(A)[0]


def rdm1_red_from_inverse(Aconj, B, M_inv, v=None):
    if v is None:
        return jnp.einsum("ki,kl,lj->ij", B, M_inv, Aconj)
    return jnp.einsum("ki,kl,lj,j...->i...", B, M_inv, Aconj, v)


def rdm1_red_and_det(Aconj, B, v=None):
    M_inv, determinant = inv_and_det(Aconj @ B.T)
    return rdm1_red_from_inverse(Aconj, B, M_inv, v=v), determinant


def rdm1_red(Aconj, B, v=None):
    # without det M
    return rdm1_red_from_inverse(Aconj, B, inv(Aconj @ B.T), v=v)


def rdm1_full(Uconj, V, v=None):
    res, determinant = rdm1_red_and_det(Uconj, V, v=v)
    return res * determinant


def E1e(Aconj, B, H1e, _red=False):
    if H1e is None:
        return 0.0
    if _red:
        r1 = rdm1_red(Aconj, B)
        determinant = None
    else:
        r1, determinant = rdm1_red_and_det(Aconj, B)

    E1_red = jnp.einsum("ba,ab", r1, H1e)

    if _red:
        return E1_red
    else:
        return determinant * E1_red


E1e_red = partial(E1e, _red=True)


def E2e(Aconj, B, H2e, _red=False):
    if H2e is None:
        return 0.0
    if _red:
        r1 = rdm1_red(Aconj, B)
        determinant = None
    else:
        r1, determinant = rdm1_red_and_det(Aconj, B)

    _, Ht, _ = antisymmetrized(H2e)
    E2_red = jnp.einsum("abge,ea,gb", Ht, r1, r1)
    if _red:
        return E2_red
    else:
        return determinant * E2_red


E2e_red = partial(E2e, _red=True)


def expval(X1c, X2, deriv_indices=None, hamiltonian=None):
    # compute <>, <H>
    # GHF version;
    # non-svd version; assumes overlap is nonzero

    ol_ = ol(X1c, X2)

    if hamiltonian is None:
        hamiltonian = {}

    H2e = hamiltonian.get('H2e', None)
    H1e = hamiltonian.get('H1e', None)
    Enuc = hamiltonian.get('Enuc', 0)

    e0_ = Enuc
    e1_ = E1e_red(X1c, X2, H1e)
    e2_ = E2e_red(X1c, X2, H2e)
    expval = ol_ * (e0_ + e1_ + e2_)
    return ol_[None, None], expval[None, None]


def expval_sq(X1c, X2, deriv_indices=None, hamiltonian=None):
    raise NotImplementedError("<H^2> has no unstable implementation; use the stable or hybrid one")
