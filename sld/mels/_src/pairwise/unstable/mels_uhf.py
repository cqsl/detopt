import jax.numpy as jnp
import numpy as np
from functools import partial

from sld.utils.hamiltonian.orb_conversion import to_desc_order_dense

from ..common import ol
from sld.utils.antisymm import Sn
from .mels import rdm1_red, E1e_red, E2e_red


def E2eud_red(X1uc, X1dc, X2u, X2d, w, mo_coeff_u, mo_coeff_d):
    if w is None:
        return 0
    r1u = rdm1_red(X1uc, X2u)
    r1d = rdm1_red(X1dc, X2d)

    if mo_coeff_u is not None:
        assert mo_coeff_d is None
        return jnp.einsum("IJKL,Ii,Jj,Kk,Ll,ij,kl", w, mo_coeff_d, mo_coeff_d, mo_coeff_u, mo_coeff_u, r1d, r1u)
    else:
        assert mo_coeff_u is None
        return jnp.einsum("ijkl,ij,kl", w, r1d, r1u)


def expval_sq_spin(X1uc, X1dc, X2u, X2d, deriv_indices=None, hamiltonian=None, mo_coeff=None, sq=True):
    # compute <>, <H^2> and <H>
    # UHF version;
    # non-svd version; assumes overlap is nonzero

    old = ol(X1dc, X2d)
    olu = ol(X1uc, X2u)
    ol_ = olu * old

    if hamiltonian is None:
        return ol_[None, None], ol_[None, None], ol_[None, None]
    else:
        H2e, H1e, Enuc, w = hamiltonian

    # remove Ht and Htt
    if isinstance(H2e, tuple):
        H2e, _, _ = H2e

    # NB: the rest of this function requires that H2e is in normal order (for historic reasons); TODO update
    if H2e is not None:
        # recompute H2e in normal order from w
        H2e = to_desc_order_dense(0.5 * w.transpose(0, 2, 3, 1), cutoff=1e-14)

    if isinstance(mo_coeff, tuple):
        mo_coeff_u, mo_coeff_d = mo_coeff
    else:
        mo_coeff_u = mo_coeff_d = mo_coeff

    eu1_ = E1e_red(X1uc, X2u, H1e, mo_coeff_u)
    ed1_ = E1e_red(X1dc, X2d, H1e, mo_coeff_d)
    eu2_ = E2e_red(X1uc, X2u, H2e, mo_coeff_u)
    ed2_ = E2e_red(X1dc, X2d, H2e, mo_coeff_d)
    eud_ = E2eud_red(X1uc, X1dc, X2u, X2d, w, mo_coeff_u, mo_coeff_d)
    expval = ol_ * (Enuc + ed1_ + ed2_ + eu1_ + eu2_ + eud_)

    if not sq:
        # match derivative interface for use in optimization
        return ol_[None, None], expval[None, None]
    else:
        if mo_coeff is not None:
            raise NotImplementedError

    # terms with Enuc

    # 00
    res00 = Enuc * Enuc
    # 02 + 04 + 20 + 40
    res024 = Enuc * (eu1_ + eu2_ + ed1_ + ed2_)
    # 0w + w0
    res0w = eud_ * Enuc

    r1u = rdm1_red(X1uc, X2u)
    r1d = rdm1_red(X1dc, X2d)

    if H1e is not None:
        # terms with H1e
        res22 = 0

        H1e = H1e.astype(r1d.dtype)  # workaround for JaxRuntimeError: INTERNAL: Unexpected GEMM dtype: f64 f64 c128 on gpu

        res22 += jnp.einsum("ix,xj,ij", H1e, H1e, r1d)
        res22 += jnp.einsum("ix,xj,ij", H1e, H1e, r1u)
        for sgn, s in Sn("kl"):
            res22 += sgn * jnp.einsum(f"i{s[0]},j{s[1]},ik,jl", H1e, H1e, r1d, r1d)
            res22 += sgn * jnp.einsum(f"i{s[0]},j{s[1]},ik,jl", H1e, H1e, r1u, r1u)

        # different spin
        res22 += 2 * jnp.einsum("ij,kl,ij,kl", H1e, H1e, r1d, r1u)  # use symmetry -> 2x
    else:
        res22 = 0
    # terms with H4

    if H2e is not None:
        res44 = 0
        # wick

        # same spin, no contraction, just normal order
        for sgn, s in Sn("klcd"):
            res44 += sgn * jnp.einsum(f"ij{s[:2]},ab{s[2:]},ik,jl,ac,bd", H2e, H2e, r1u, r1u, r1u, r1u)
            res44 += sgn * jnp.einsum(f"ij{s[:2]},ab{s[2:]},ik,jl,ac,bd", H2e, H2e, r1d, r1d, r1d, r1d)

        # same spin, single contraction
        for i, j in list(np.ndindex(2, 2)):
            s1 = ["ijxu", "ijux"][i]
            s2 = ["ukyz", "kuyz"][j]
            for sgn, s in Sn("xyz"):
                st = f"{s1},{s2},i{s[0]},j{s[1]},k{s[2]}"
                sgn = sgn * (-1) ** (i + j)
                res44 += sgn * jnp.einsum(st, H2e, H2e, r1u, r1u, r1u)
                res44 += sgn * jnp.einsum(st, H2e, H2e, r1d, r1d, r1d)

        # # same spin, double contraction
        # we only need one, the other is zero because of the symmetry in H
        for sgn, s in Sn("kl"):
            res44 += sgn * jnp.einsum(f"ijab,abkl,i{s[0]},j{s[1]}", H2e, H2e, r1u, r1u)
            res44 += sgn * jnp.einsum(f"ijab,abkl,i{s[0]},j{s[1]}", H2e, H2e, r1d, r1d)

        # different spin
        res44 += 2 * ed2_ * eu2_
    else:
        res44 = 0

    # terms with Hw @ Hw
    # we need again wick

    if w is not None:
        resww = 0

        # no contractions
        for sgn1, s1 in Sn("kl"):
            for sgn2, s2 in Sn("cd"):
                st = f"ikac,jlbd,i{s1[0]},j{s1[1]},a{s2[0]},b{s2[1]}"
                resww += sgn1 * sgn2 * jnp.einsum(st, w, w, r1u, r1u, r1d, r1d)  # use symmetry of w

        # one contraction of one  of them
        for sgn1, s1 in Sn("kl"):
            st = f"ikac,jlcd,i{s1[0]},j{s1[1]},ad"
            resww += sgn1 * jnp.einsum(st, w, w, r1u, r1u, r1d)

        for sgn2, s2 in Sn("cd"):
            st = f"ikac,klbd,il,a{s2[0]},b{s2[1]}"
            resww += sgn2 * jnp.einsum(st, w, w, r1u, r1d, r1d)

        # one contraction each
        st = "ikac,klcd,il,ad"
        resww += jnp.einsum(st, w, w, r1u, r1d)  # use symmetry of w
    else:
        resww = 0

    # terms with H4 @ H1e / H1e @ H4
    if H1e is not None and H2e is not None:
        res42 = 0
        # wick

        # same spin, no contraction, just normal order
        for sgn, s in Sn("klc"):
            res42 -= sgn * jnp.einsum(f"ij{s[:2]},a{s[2:]},ik,jl,ac", H2e, H1e, r1u, r1u, r1u)
            res42 -= sgn * jnp.einsum(f"ij{s[:2]},a{s[2:]},ik,jl,ac", H2e, H1e, r1d, r1d, r1d)

        # # different spin
        res42 += eu2_ * ed1_ + ed2_ * eu1_

        res24 = res42.copy()

        # # same spin, single contraction
        for i in range(2):
            s1 = ["ijxu", "ijux"][i]
            s2 = "uy"
            for sgn, s in Sn("xy"):
                st = f"{s1},{s2},{s[0]}i,{s[1]}j"
                sgn = sgn * (-1) ** i
                res42 -= sgn * jnp.einsum(st, H2e, H1e, r1u, r1u)
                res42 -= sgn * jnp.einsum(st, H2e, H1e, r1d, r1d)

        # # same spin, single contraction
        for i in range(2):
            s1 = ["xuij", "uxij"][i]
            s2 = "yu"
            for sgn, s in Sn("xy"):
                st = f"{s1},{s2},i{s[0]},j{s[1]}"
                sgn = sgn * (-1) ** i
                res24 -= sgn * jnp.einsum(st, H2e, H1e, r1u, r1u)
                res24 -= sgn * jnp.einsum(st, H2e, H1e, r1d, r1d)
    else:
        res42 = res24 = 0

    # terms with H @ H1e / H1e @ w
    # we need again wick

    if w is not None and H1e is not None:
        res2w = 0

        # no contractions

        for sgn1, s1 in Sn("kl"):
            st = f"ikab,jl,i{s1[0]},j{s1[1]},ab"
            res2w += sgn1 * jnp.einsum(st, w, H1e, r1u, r1u, r1d)

        for sgn1, s1 in Sn("kl"):
            st = f"ikab,jl,i{s1[0]},j{s1[1]},ab"
            res2w += sgn1 * jnp.einsum(st, w, H1e, r1d, r1d, r1u)

        # single contraction

        resw2 = res2w.copy()

        st = "ikab,kl,il,ab"
        res2w += jnp.einsum(st, w, H1e, r1u, r1d)
        res2w += jnp.einsum(st, w, H1e, r1d, r1u)

        st = "klab,ik,il,ab"
        resw2 += jnp.einsum(st, w, H1e, r1u, r1d)
        resw2 += jnp.einsum(st, w, H1e, r1d, r1u)
    else:
        res2w = resw2 = 0

    # terms with Hw @ H4 / H4 @ Hw
    if H2e is not None and w is not None:
        res4w = 0

        # no contractions
        for sgn, s in Sn("def"):
            st = f"ijad,bcef,ij,a{s[0]},b{s[1]},c{s[2]}"
            res4w -= sgn * jnp.einsum(st, w, H2e, r1u, r1d, r1d, r1d)
            res4w -= sgn * jnp.einsum(st, w, H2e, r1d, r1u, r1u, r1u)

        resw4 = res4w.copy()

        # single contraction
        for sgn, s in Sn("ef"):
            st = f"ijac,bcef,ij,a{s[0]},b{s[1]}"
            res4w += sgn * jnp.einsum(st, w, H2e, r1u, r1d, r1d)
            res4w += sgn * jnp.einsum(st, w, H2e, r1d, r1u, r1u)
            st = f"ijac,cbef,ij,a{s[0]},b{s[1]}"
            res4w -= sgn * jnp.einsum(st, w, H2e, r1u, r1d, r1d)
            res4w -= sgn * jnp.einsum(st, w, H2e, r1d, r1u, r1u)

        # single contraction
        for sgn, s in Sn("df"):
            st = f"ijxd,bcxf,ij,b{s[0]},c{s[1]}"
            resw4 -= sgn * jnp.einsum(st, w, H2e, r1u, r1d, r1d)
            resw4 -= sgn * jnp.einsum(st, w, H2e, r1d, r1u, r1u)
            st = f"ijxd,bcfx,ij,b{s[0]},c{s[1]}"
            resw4 += sgn * jnp.einsum(st, w, H2e, r1u, r1d, r1d)
            resw4 += sgn * jnp.einsum(st, w, H2e, r1d, r1u, r1u)
    else:
        res4w = resw4 = 0

    expval_sq = 0
    expval_sq += res00 + 2 * res024 + 2 * res0w
    expval_sq += res22 + res24 + res42 + res2w + resw2
    expval_sq += res44 + res4w + resw4
    expval_sq += resww

    expval_sq = expval_sq * ol_

    return ol_[None, None], expval_sq[None, None], expval[None, None]


expval_spin = partial(expval_sq_spin, sq=False)
