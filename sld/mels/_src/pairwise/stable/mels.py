import jax.numpy as jnp

import numpy as np
from functools import partial

from ..common import ol
from .common import T, A_inv_det_stable_1
from sld.utils.antisymm import Sn


def rdm1_full(Uconj, V):
    A = Uconj @ V.T
    detA, A_inv = A_inv_det_stable_1(A)
    # transposed, because its rho_ab, with Ua and Vb
    return Uconj.T @ A_inv.T @ V * detA


def E1e(X1c, X2, H1e, mo_coeff=None):
    if H1e is None:
        return 0

    D, t1, U, V = T(X1c, X2, 1)
    if mo_coeff is not None:
        return jnp.einsum("IJ,Ii,Jj,io,oj,o", H1e, mo_coeff, mo_coeff, U, V, t1) * D
    else:
        return jnp.einsum("ij,io,oj,o", H1e, U, V, t1) * D


def E2e(X1c, X2, H2e, mo_coeff=None):
    if H2e is None:
        return 0

    D, t2, U, V = T(X1c, X2, 2)

    e2 = 0
    for sgn, s in Sn("kl"):
        if mo_coeff is not None:
            # meaning of capital letters is reversed here
            e2 -= sgn * jnp.einsum(f"ij{s},iI,jJ,kK,lL,IO,OK,JP,PL,OP", H2e, mo_coeff, mo_coeff, mo_coeff, mo_coeff, U, V, U, V, t2) * D
        else:
            e2 -= sgn * jnp.einsum(f"ij{s},io, ok,jp, pl, op", H2e, U, V, U, V, t2) * D
    return e2


def E1e_sq(X1c, X2, H1e, mo_coeff=None):
    if H1e is None:
        return 0

    if mo_coeff is not None:
        raise NotImplementedError
    D, t1, U, V = T(X1c, X2, 1)
    D, t2, U, V = T(X1c, X2, 2)

    res22 = 0
    res22 += jnp.einsum("ix,xj,io, oj, o", H1e, H1e, U, V, t1) * D
    for sgn, s in Sn("kl"):
        res22 += sgn * jnp.einsum(f"i{s[0]},j{s[1]},io, ok,jp, pl, op", H1e, H1e, U, V, U, V, t2) * D
    return res22


def E2e_sq(X1c, X2, H2e, mo_coeff=None):
    if H2e is None:
        return 0

    if mo_coeff is not None:
        raise NotImplementedError

    D, t2, U, V = T(X1c, X2, 2)
    D, t3, U, V = T(X1c, X2, 3)
    D, t4, U, V = T(X1c, X2, 4)

    # terms with H2e
    res44 = 0
    # no contraction, just normal order
    for sgn, s in Sn("klcd"):
        res44 += sgn * jnp.einsum(f"ij{s[:2]},ab{s[2:]},io,ok,jp,pl,aq,qc,br,rd, opqr", H2e, H2e, U, V, U, V, U, V, U, V, t4) * D

    # single contraction
    for i, j in list(np.ndindex(2, 2)):
        s1 = ["ijxu", "ijux"][i]
        s2 = ["ukyz", "kuyz"][j]
        for sgn, s in Sn("xyz"):
            st = f"{s1},{s2},io, o{s[0]},jp, p{s[1]},kq, q{s[2]}, opq"
            sgn = sgn * (-1) ** (i + j)
            res44 += sgn * jnp.einsum(st, H2e, H2e, U, V, U, V, U, V, t3) * D

    # double contraction
    # we only need one, the other is zero because of the symmetry in H
    for sgn, s in Sn("kl"):
        res44 += sgn * jnp.einsum(f"ijab,abkl,io,o{s[0]},jp,p{s[1]},op", H2e, H2e, U, V, U, V, t2) * D
    return res44


def E1e2e(X1c, X2, H1e, H2e, mo_coeff=None, _swap=False):
    if H1e is None or H2e is None:
        return 0

    if mo_coeff is not None:
        raise NotImplementedError

    D, t2, U, V = T(X1c, X2, 2)
    D, t3, U, V = T(X1c, X2, 3)

    res42 = 0
    # no contraction, just normal order
    for sgn, s in Sn("klc"):
        res42 -= sgn * jnp.einsum(f"ij{s[:2]},a{s[2:]},io,ok,jp,pl,aq,qc,opq", H2e, H1e, U, V, U, V, U, V, t3) * D

    res24 = res42.copy()

    # single contraction
    for i in range(2):
        s1 = ["ijxu", "ijux"][i]
        s2 = "uy"
        for sgn, s in Sn("xy"):
            st = f"{s1},{s2},{s[0]}o,oi,{s[1]}p,pj,op"
            sgn = sgn * (-1) ** i
            res42 -= sgn * jnp.einsum(st, H2e, H1e, U, V, U, V, t2) * D

    for i in range(2):
        s1 = ["xuij", "uxij"][i]
        s2 = "yu"
        for sgn, s in Sn("xy"):
            st = f"{s1},{s2},ip,p{s[0]},jq,q{s[1]},pq"
            sgn = sgn * (-1) ** i
            res24 -= sgn * jnp.einsum(st, H2e, H1e, U, V, U, V, t2) * D

    if _swap:
        return res42
    else:
        return res24


E2e1e = partial(E1e2e, _swap=True)


def expval_sq(X1c, X2, deriv_indices=None, hamiltonian=None, mo_coeff=None, _sq=True):
    # compute <>, <H^2> and <H>
    # GHF version;
    # svd version; works with zero overlap

    ol_ = ol(X1c, X2)

    if hamiltonian is None:
        if _sq:
            return ol_[None, None], ol_[None, None], ol_[None, None]
        else:
            return ol_[None, None], ol_[None, None]
    else:
        H2e, H1e, Enuc = hamiltonian

    if isinstance(H2e, tuple):
        H2e, _, _ = H2e

    e1 = E1e(X1c, X2, H1e, mo_coeff)
    e2 = E2e(X1c, X2, H2e, mo_coeff)
    expval = Enuc * ol_ + e1 + e2

    if not _sq:
        # match derivative interface for use in optimization
        return ol_[None, None], expval[None, None]

    if mo_coeff is not None:
        raise NotImplementedError  # TODO

    # terms with Enuc

    # 00
    res00 = Enuc * Enuc * ol_
    # 02 + 04 + 20 + 40
    res024 = Enuc * (e1 + e2)

    res22 = E1e_sq(X1c, X2, H1e, mo_coeff)
    res44 = E2e_sq(X1c, X2, H2e, mo_coeff)

    res24 = E1e2e(X1c, X2, H1e, H2e, mo_coeff)
    res42 = E2e1e(X1c, X2, H1e, H2e, mo_coeff)

    expval_sq = 0
    expval_sq += res00 + 2 * res024
    expval_sq += res22 + res24 + res42
    expval_sq += res44

    return ol_[None, None], expval_sq[None, None], expval[None, None]


expval = partial(expval_sq, _sq=False)
