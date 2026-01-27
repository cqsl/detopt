import jax.numpy as jnp

from functools import partial

from sld.utils.hamiltonian.orb_conversion import to_desc_order_dense
from ..common import ol
from .common import T
from sld.utils.antisymm import Sn

from .mels import E1e, E2e, E1e_sq, E2e_sq


def E2eud(X1uc, X2u, X1dc, X2d, w, mo_coeff_u, mo_coeff_d):
    if w is None:
        return 0
    Du, t1u, Uu, Vu = T(X1uc, X2u, 1)
    Dd, t1d, Ud, Vd = T(X1dc, X2d, 1)

    if mo_coeff_u is not None:
        eud = jnp.einsum("IJKL,Ii,Jj,Kk,Ll,io,oj,o,kp,pl,p", w, mo_coeff_d, mo_coeff_d, mo_coeff_u, mo_coeff_u, Ud, Vd, t1d, Uu, Vu, t1u) * Dd * Du
    else:
        eud = jnp.einsum("ijkl,io,oj,o,kp,pl,p", w, Ud, Vd, t1d, Uu, Vu, t1u) * Dd * Du
    return eud


def E2eud_sq(X1uc, X2u, X1dc, X2d, w, mo_coeff_u, mo_coeff_d):
    # terms with Hw @ Hw
    # we need again wick
    if w is None:
        return 0

    if mo_coeff_u is not None or mo_coeff_d is not None:
        raise NotImplementedError

    Du, t1u, Uu, Vu = T(X1uc, X2u, 1)
    Dd, t1d, Ud, Vd = T(X1dc, X2d, 1)
    Du, t2u, Uu, Vu = T(X1uc, X2u, 2)
    Dd, t2d, Ud, Vd = T(X1dc, X2d, 2)

    resww = 0

    # no contractions
    for sgn1, s1 in Sn("kl"):
        for sgn2, s2 in Sn("cd"):
            st = f"ikac,jlbd,io, o{s1[0]},jp, p{s1[1]},op, aq, q{s2[0]},br, r{s2[1]}, qr"
            resww += sgn1 * sgn2 * jnp.einsum(st, w, w, Uu, Vu, Uu, Vu, t2u, Ud, Vd, Ud, Vd, t2d) * Du * Dd  # use symmetry of w

    # one contraction of one  of them
    for sgn1, s1 in Sn("kl"):
        st = f"ikac,jlcd,io,o{s1[0]},jp, p{s1[1]}, op,aq, qd, q"
        resww += sgn1 * jnp.einsum(st, w, w, Uu, Vu, Uu, Vu, t2u, Ud, Vd, t1d) * Du * Dd

    for sgn2, s2 in Sn("cd"):
        st = f"ikac,klbd,io,ol, o,ap, p{s2[0]},bq, q{s2[1]}, pq"
        resww += sgn2 * jnp.einsum(st, w, w, Uu, Vu, t1u, Ud, Vd, Ud, Vd, t2d) * Du * Dd

    # one contraction each
    st = "ikac,klcd,io, ol,o, ap,pd, p"
    resww += jnp.einsum(st, w, w, Uu, Vu, t1u, Ud, Vd, t1d) * Du * Dd  # use symmetry of w
    return resww


def expval_sq_spin(X1uc, X1dc, X2u, X2d, deriv_indices=None, hamiltonian=None, mo_coeff=None, _sq=True):
    # compute <>, <H^2> and <H>
    # UHF version;
    # svd version; works with zero overlap

    if isinstance(mo_coeff, tuple):
        mo_coeff_u, mo_coeff_d = mo_coeff
    else:
        mo_coeff_u = mo_coeff_d = mo_coeff

    old = ol(X1dc, X2d)
    olu = ol(X1uc, X2u)
    ol_ = olu * old

    if hamiltonian is None:
        return (ol_[None, None],) * (3 if _sq else 2)
    else:
        H2e, H1e, Enuc, w = hamiltonian

    # remove Ht and Htt
    if isinstance(H2e, tuple):
        H2e, _, _ = H2e

    # NB: the rest of this function requires that H2e is in normal order (for historic reasons); TODO update
    if H2e is not None:
        # recompute H2e in normal order from w
        assert w is not None
        H2e = to_desc_order_dense(0.5 * w.transpose(0, 2, 3, 1), cutoff=1e-14)

    Du, t1u, Uu, Vu = T(X1uc, X2u, 1)
    Dd, t1d, Ud, Vd = T(X1dc, X2d, 1)
    Du, t2u, Uu, Vu = T(X1uc, X2u, 2)
    Dd, t2d, Ud, Vd = T(X1dc, X2d, 2)
    Du, t3u, Uu, Vu = T(X1uc, X2u, 3)
    Dd, t3d, Ud, Vd = T(X1dc, X2d, 3)
    Du, t4u, Uu, Vu = T(X1uc, X2u, 4)
    Dd, t4d, Ud, Vd = T(X1dc, X2d, 4)

    eu1_ = E1e(X1uc, X2u, H1e, mo_coeff=mo_coeff_u)
    ed1_ = E1e(X1dc, X2d, H1e, mo_coeff=mo_coeff_d)
    eu2_ = E2e(X1uc, X2u, H2e, mo_coeff=mo_coeff_u)
    ed2_ = E2e(X1dc, X2d, H2e, mo_coeff=mo_coeff_d)

    eu1 = eu1_ * old
    ed1 = ed1_ * olu
    eu2 = eu2_ * old
    ed2 = ed2_ * olu

    eud = E2eud(X1uc, X2u, X1dc, X2d, w, mo_coeff_u, mo_coeff_d)

    expval = Enuc * old * olu + ed1 + ed2 + eu1 + eu2 + eud

    if not _sq:
        # match derivative interface for use in optimization
        return (old * olu)[None, None], expval[None, None]
    else:
        if mo_coeff is not None:
            raise NotImplementedError

    # terms with Enuc

    # 00
    res00 = Enuc * Enuc * old * olu
    # 02 + 04 + 20 + 40
    res024 = Enuc * (eu1 + eu2 + ed1 + ed2)
    # 0w + w0
    res0w = eud * Enuc

    # terms with H1e
    res22 = 0
    # same spin
    res22 += E1e_sq(X1uc, X2u, H1e, mo_coeff_u) * old
    res22 += E1e_sq(X1dc, X2d, H1e, mo_coeff_d) * olu
    # different spin
    res22 += 2 * eu1_ * ed1_  # use symmetry -> 2x

    # terms with H4
    res44 = 0
    # same spin
    res22 += E2e_sq(X1uc, X2u, H2e, mo_coeff_u) * old
    res22 += E2e_sq(X1dc, X2d, H2e, mo_coeff_d) * olu
    # different spin
    res44 += 2 * ed2_ * eu2_  # use symmetry -> 2x

    # terms with Hw @ Hw
    resww = E2eud_sq(X1uc, X2u, X1dc, X2d, w, mo_coeff_u, mo_coeff_d)

    # terms with H4 @ H1e / H1e @ H4
    if H1e is not None and H2e is not None:
        res42 = 0
        # wick

        # same spin, no contraction, just normal order
        for sgn, s in Sn("klc"):
            res42 -= sgn * jnp.einsum(f"ij{s[:2]},a{s[2:]},io,ok,jp,pl,aq,qc,opq", H2e, H1e, Uu, Vu, Uu, Vu, Uu, Vu, t3u) * Du * old
            res42 -= sgn * jnp.einsum(f"ij{s[:2]},a{s[2:]},io,ok,jp,pl,aq,qc,opq", H2e, H1e, Ud, Vd, Ud, Vd, Ud, Vd, t3d) * Dd * olu

        # # different spin
        res42 += eu2_ * ed1_ + ed2_ * eu1_

        res24 = res42.copy()

        # # same spin, single contraction
        for i in range(2):
            s1 = ["ijxu", "ijux"][i]
            s2 = "uy"
            for sgn, s in Sn("xy"):
                st = f"{s1},{s2},{s[0]}o,oi,{s[1]}p,pj,op"
                sgn = sgn * (-1) ** i
                res42 -= sgn * jnp.einsum(st, H2e, H1e, Uu, Vu, Uu, Vu, t2u) * Du * old
                res42 -= sgn * jnp.einsum(st, H2e, H1e, Ud, Vd, Ud, Vd, t2d) * Dd * olu

        # # same spin, single contraction
        for i in range(2):
            s1 = ["xuij", "uxij"][i]
            s2 = "yu"
            for sgn, s in Sn("xy"):
                st = f"{s1},{s2},ip,p{s[0]},jq,q{s[1]},pq"
                sgn = sgn * (-1) ** i
                res24 -= sgn * jnp.einsum(st, H2e, H1e, Uu, Vu, Uu, Vu, t2u) * Du * old
                res24 -= sgn * jnp.einsum(st, H2e, H1e, Ud, Vd, Ud, Vd, t2d) * Dd * olu
    else:
        res42 = res24 = 0

    # terms with H @ H1e / H1e @ w
    # we need again wick

    if w is not None and H1e is not None:
        res2w = 0

        # no contractions

        for sgn1, s1 in Sn("kl"):
            st = f"ikab,jl,io,o{s1[0]},jp,p{s1[1]},op, aq,qb,q"
            res2w += sgn1 * jnp.einsum(st, w, H1e, Uu, Vu, Uu, Vu, t2u, Ud, Vd, t1d) * Du * Dd

        for sgn1, s1 in Sn("kl"):
            st = f"ikab,jl,io,o{s1[0]},jp,p{s1[1]},op,aq,qb,q"
            res2w += sgn1 * jnp.einsum(st, w, H1e, Ud, Vd, Ud, Vd, t2d, Uu, Vu, t1u) * Du * Dd

        # single contraction

        resw2 = res2w.copy()

        st = "ikab,kl,io,ol,o,ap,pb,p"
        res2w += jnp.einsum(st, w, H1e, Uu, Vu, t1u, Ud, Vd, t1d) * Du * Dd
        res2w += jnp.einsum(st, w, H1e, Ud, Vd, t1d, Uu, Vu, t1u) * Du * Dd

        st = "klab,ik,io,ol,o,ap,pb,p"
        resw2 += jnp.einsum(st, w, H1e, Uu, Vu, t1u, Ud, Vd, t1d) * Du * Dd
        resw2 += jnp.einsum(st, w, H1e, Ud, Vd, t1d, Uu, Vu, t1u) * Du * Dd
    else:
        res2w = resw2 = 0

    # terms with Hw @ H4 / H4 @ Hw
    if H2e is not None and w is not None:
        res4w = 0

        # no contractions
        for sgn, s in Sn("def"):
            st = f"ijad,bcef,io,oj,o,ap,p{s[0]},bq,q{s[1]},cr,r{s[2]},pqr"
            res4w -= sgn * jnp.einsum(st, w, H2e, Uu, Vu, t1u, Ud, Vd, Ud, Vd, Ud, Vd, t3d) * Du * Dd
            res4w -= sgn * jnp.einsum(st, w, H2e, Ud, Vd, t1d, Uu, Vu, Uu, Vu, Uu, Vu, t3u) * Du * Dd

        resw4 = res4w.copy()

        # single contraction
        for sgn, s in Sn("ef"):
            st = f"ijac,bcef,io,oj,o,ap,p{s[0]},bq,q{s[1]},pq"
            res4w += sgn * jnp.einsum(st, w, H2e, Uu, Vu, t1u, Ud, Vd, Ud, Vd, t2d) * Du * Dd
            res4w += sgn * jnp.einsum(st, w, H2e, Ud, Vd, t1d, Uu, Vu, Uu, Vu, t2u) * Du * Dd
            st = f"ijac,cbef,io,oj,o,ap,p{s[0]},bq,q{s[1]},pq"
            res4w -= sgn * jnp.einsum(st, w, H2e, Uu, Vu, t1u, Ud, Vd, Ud, Vd, t2d) * Du * Dd
            res4w -= sgn * jnp.einsum(st, w, H2e, Ud, Vd, t1d, Uu, Vu, Uu, Vu, t2u) * Du * Dd

        # single contraction
        for sgn, s in Sn("df"):
            st = f"ijxd,bcxf,io,oj,o,bp,p{s[0]},cq,q{s[1]},pq"
            resw4 -= sgn * jnp.einsum(st, w, H2e, Uu, Vu, t1u, Ud, Vd, Ud, Vd, t2d) * Du * Dd
            resw4 -= sgn * jnp.einsum(st, w, H2e, Ud, Vd, t1d, Uu, Vu, Uu, Vu, t2u) * Du * Dd
            st = f"ijxd,bcfx,io,oj,o,bp,p{s[0]},cq,q{s[1]},pq"
            resw4 += sgn * jnp.einsum(st, w, H2e, Uu, Vu, t1u, Ud, Vd, Ud, Vd, t2d) * Du * Dd
            resw4 += sgn * jnp.einsum(st, w, H2e, Ud, Vd, t1d, Uu, Vu, Uu, Vu, t2u) * Du * Dd
    else:
        res4w = resw4 = 0

    expval_sq = 0
    expval_sq += res00 + 2 * res024 + 2 * res0w
    expval_sq += res22 + res24 + res42 + res2w + resw2
    expval_sq += res44 + res4w + resw4
    expval_sq += resww

    return (olu * old)[None, None], expval_sq[None, None], expval[None, None]


expval_spin = partial(expval_sq_spin, _sq=False)
