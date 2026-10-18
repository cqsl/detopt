import jax.numpy as jnp

from functools import partial

from ..common import ol, antisymmetrized
from .common import T, A_inv_det_stable_1
from sld.utils.antisymm import Sn


def rdm1_full(Uconj, V):
    A = Uconj @ V.T
    detA, A_inv = A_inv_det_stable_1(A)
    return V.T @ A_inv @ Uconj * detA


def E1e(X1c, X2, H1e):
    if H1e is None:
        return 0

    D, t1, U, V = T(X1c, X2, 1)
    return jnp.einsum("ij,jo,oi,o", H1e, U, V, t1) * D


def E2e(X1c, X2, H2e):
    if H2e is None:
        return 0
    _, _, Htt = antisymmetrized(H2e)

    H2e_bra = jnp.einsum("ijkl,ai,bj->abkl", Htt, X1c, X1c, optimize="optimal")
    H2e_occ = jnp.einsum("abkl,ck,dl->abcd", H2e_bra, X2, X2, optimize="optimal")

    D, t2, _, _, X, YH = T(X1c, X2, 2, return_svd=True)
    left = X.T.conj()
    right = YH.T.conj()
    return -0.5 * D * jnp.einsum(
        "abcd,oa,co,pb,dp,op", H2e_occ, left, right, left, right, t2, optimize="optimal"
    )


def E1e_sq(X1c, X2, H1e):
    if H1e is None:
        return 0

    D, t1, U, V = T(X1c, X2, 1)
    D, t2, U, V = T(X1c, X2, 2)

    res22 = 0
    res22 += jnp.einsum("ix,xj,io, oj, o", H1e, H1e, U, V, t1, optimize="optimal") * D
    for sgn, s in Sn("kl"):
        res22 += sgn * jnp.einsum(f"i{s[0]},j{s[1]},io, ok,jp, pl, op", H1e, H1e, U, V, U, V, t2, optimize="optimal") * D
    return res22


def E2e_sq(X1c, X2, H2e):
    if H2e is None:
        return 0
    _, _, Htt = antisymmetrized(H2e)

    D, t2, U, V = T(X1c, X2, 2)
    _, t3, _, _ = T(X1c, X2, 3)
    _, t4, _, _ = T(X1c, X2, 4)

    Ptt = jnp.einsum("ijkl,io,jp->opkl", Htt, U, U, optimize="optimal")
    Qtt = jnp.einsum("ukyz,kq->uqyz", Htt, U, optimize="optimal")

    X, _, _ = jnp.linalg.svd(X1c @ X2.T)
    left = X.T.conj()
    Rtt_bra = jnp.einsum("abkl,ek,fl->abef", Htt, X1c, X1c, optimize="optimal")
    Rtt = jnp.einsum("abef,oe,pf->abop", Rtt_bra, left, left, optimize="optimal")

    G2 = jnp.einsum("opkl,ok,pl->op", Ptt, V, V, optimize="optimal")
    G4 = jnp.einsum("opkl,qk,rl->opqr", Ptt, V, V, optimize="optimal")
    Att = jnp.einsum("opkc,ok,qc->opq", Ptt, V, V, optimize="optimal")
    Btt = jnp.einsum("qrld,pl,rd->qpr", Ptt, V, V, optimize="optimal")

    # terms with H2e
    res44 = 0
    # no contraction
    res44 += 0.25 * jnp.einsum("op,qr,opqr", G2, G2, t4, optimize="optimal") * D
    res44 += 0.25 * jnp.einsum("opqr,qrop,opqr", G4, G4, t4, optimize="optimal") * D
    res44 -= jnp.einsum("opq,qpr,opqr", Att, Btt, t4, optimize="optimal") * D

    # single contraction
    C = jnp.einsum("opxu,ox->opu", Ptt, V, optimize="optimal")
    Cx = 0.5 * jnp.einsum("opxu,qx->opqu", Ptt, V, optimize="optimal")
    Dc = jnp.einsum("uqyz,qz,py->uqp", Qtt, V, V, optimize="optimal")
    Dx = jnp.einsum("uqyz,oy,pz->uqop", Qtt, V, V, optimize="optimal")
    res44 += jnp.einsum("opu,uqp,opq", C, Dc, t3, optimize="optimal") * D
    res44 += jnp.einsum("opqu,uqop,opq", Cx, Dx, t3, optimize="optimal") * D

    # double contraction
    res44 += 0.25 * jnp.einsum("opab,abop,op", Ptt, Rtt, t2, optimize="optimal") * D
    return res44


def E1e2e_both(X1c, X2, H1e, H2e):
    if H1e is None or H2e is None:
        return 0, 0
    _, _, Htt = antisymmetrized(H2e)

    D, t2, U, V = T(X1c, X2, 2)
    _, t3, _, _ = T(X1c, X2, 3)

    Ptt = jnp.einsum("ijkl,io,jp->opkl", Htt, U, U, optimize="optimal")
    G2 = jnp.einsum("opkl,ok,pl->op", Ptt, V, V, optimize="optimal")
    Att = jnp.einsum("opkc,ok,qc->opq", Ptt, V, V, optimize="optimal")

    res42 = 0
    # no contraction
    h1_trace = jnp.einsum("ac,aq,qc->q", H1e, U, V, optimize="optimal")
    h1_open = jnp.einsum("al,aq,pl->qp", H1e, U, V, optimize="optimal")
    res42 -= 0.5 * jnp.einsum("op,q,opq", G2, h1_trace, t3, optimize="optimal") * D
    res42 += jnp.einsum("opq,qp,opq", Att, h1_open, t3, optimize="optimal") * D

    res24 = res42.copy()

    X, _, YH = jnp.linalg.svd(X1c @ X2.T)
    left, right = X.T.conj(), YH.T.conj()

    Htt_bra42 = jnp.einsum("ijxu,uy,ai,bj->abxy", Htt, H1e, X1c, X1c, optimize="optimal")
    Htt_occ42 = jnp.einsum("abxy,cx,dy->abcd", Htt_bra42, X2, X2, optimize="optimal")
    res42 -= jnp.einsum("abcd,oa,co,pb,dp,op", Htt_occ42, left, right, left, right, t2, optimize="optimal") * D

    Htt_bra24 = jnp.einsum("xuij,yu,ex,fy->efij", Htt, H1e, X1c, X1c, optimize="optimal")
    Htt_occ24 = jnp.einsum("efij,ci,dj->efcd", Htt_bra24, X2, X2, optimize="optimal")
    res24 -= jnp.einsum("efcd,oe,co,pf,dp,op", Htt_occ24, left, right, left, right, t2, optimize="optimal") * D

    return res24, res42


def expval_sq(X1c, X2, deriv_indices=None, hamiltonian=None, _sq=True):
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
        H2e = hamiltonian.get('H2e', None)
        H1e = hamiltonian.get('H1e', None)
        Enuc = hamiltonian.get('Enuc', 0)
        U = hamiltonian.get('U', None)
        if U is not None:
            raise NotImplementedError("U operators in the stable GHF matrix elements")

    e1 = E1e(X1c, X2, H1e)
    e2 = E2e(X1c, X2, H2e)
    expval = Enuc * ol_ + e1 + e2

    if not _sq:
        # match derivative interface for use in optimization
        return ol_[None, None], expval[None, None]

    # terms with Enuc

    # 00
    res00 = Enuc * Enuc * ol_
    # 02 + 04 + 20 + 40
    res024 = Enuc * (e1 + e2)

    res22 = E1e_sq(X1c, X2, H1e)
    res44 = E2e_sq(X1c, X2, H2e)

    res24, res42 = E1e2e_both(X1c, X2, H1e, H2e)

    expval_sq = 0
    expval_sq += res00 + 2 * res024
    expval_sq += res22 + res24 + res42
    expval_sq += res44

    return ol_[None, None], expval_sq[None, None], expval[None, None]


expval = partial(expval_sq, _sq=False)
