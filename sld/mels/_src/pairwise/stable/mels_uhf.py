import jax.numpy as jnp

from functools import partial

from ..common import ol, antisymmetrized
from .common import T

from .mels import E1e, E2e, E1e_sq, E2e_sq, E1e2e_both


def E2eud(X1uc, X2u, X1dc, X2d, w):
    if w is None:
        return 0
    Du, t1u, _, _ = T(X1uc, X2u, 1)
    Dd, t1d, _, _ = T(X1dc, X2d, 1)

    w_bra = jnp.einsum("ijkl,aj,bl->abik", w, X1dc, X1uc, optimize="optimal")
    w_occ = jnp.einsum("abik,ci,dk->abcd", w_bra, X2d, X2u, optimize="optimal")

    _, _, _, _, Xd, YHd = T(X1dc, X2d, 1, return_svd=True)
    _, _, _, _, Xu, YHu = T(X1uc, X2u, 1, return_svd=True)
    left_d, right_d = Xd.T.conj(), YHd.T.conj()
    left_u, right_u = Xu.T.conj(), YHu.T.conj()

    return jnp.einsum(
        "abcd,oa,co,pb,dp,o,p", w_occ, left_d, right_d, left_u, right_u, t1d, t1u, optimize="optimal"
    ) * Dd * Du


def Eonsiteud(X1uc, X2u, X1dc, X2d, U_onsite):
    Du, t1u, Uu, Vu = T(X1uc, X2u, 1)
    Dd, t1d, Ud, Vd = T(X1dc, X2d, 1)
    return U_onsite * jnp.einsum("io,oi,o,ip,pi,p", Ud, Vd, t1d, Uu, Vu, t1u) * Dd * Du


def Eonsiteud_sq(X1uc, X2u, X1dc, X2d, U_onsite):
    Du, t1u, Uu, Vu = T(X1uc, X2u, 1)
    Dd, t1d, Ud, Vd = T(X1dc, X2d, 1)
    Du, t2u, Uu, Vu = T(X1uc, X2u, 2)
    Dd, t2d, Ud, Vd = T(X1dc, X2d, 2)
    res = 0
    P = U_onsite * jnp.einsum("ko,kq,ok,qk->oq", Uu, Ud, Vu, Vd, optimize="optimal")
    Q = U_onsite * jnp.einsum("ko,kq,pk,qk->opq", Uu, Ud, Vu, Vd, optimize="optimal")
    R = U_onsite * jnp.einsum("ko,kq,ok,rk->oqr", Uu, Ud, Vu, Vd, optimize="optimal")
    S = U_onsite * jnp.einsum("ko,kq,pk,rk->opqr", Uu, Ud, Vu, Vd, optimize="optimal")
    res += jnp.einsum("oq,pr,op,qr", P, P, t2u, t2d, optimize="optimal") * Du * Dd
    res -= jnp.einsum("opq,por,op,qr", Q, Q, t2u, t2d, optimize="optimal") * Du * Dd
    res -= jnp.einsum("oqr,prq,op,qr", R, R, t2u, t2d, optimize="optimal") * Du * Dd
    res += jnp.einsum("opqr,porq,op,qr", S, S, t2u, t2d, optimize="optimal") * Du * Dd

    M1 = U_onsite * jnp.einsum("co,cq,oc->oqc", Uu, Ud, Vu, optimize="optimal")
    M2 = U_onsite * jnp.einsum("cp,qc,pc->pcq", Uu, Vd, Vu, optimize="optimal")
    M3 = U_onsite * jnp.einsum("co,cq,pc->opqc", Uu, Ud, Vu, optimize="optimal")
    M4 = U_onsite * jnp.einsum("cp,qc,oc->pocq", Uu, Vd, Vu, optimize="optimal")
    res += jnp.einsum("oqc,pcq,op,q", M1, M2, t2u, t1d, optimize="optimal") * Du * Dd
    res -= jnp.einsum("opqc,pocq,op,q", M3, M4, t2u, t1d, optimize="optimal") * Du * Dd

    N1 = U_onsite * jnp.einsum("ko,kp,pk->okp", Uu, Ud, Vd, optimize="optimal")
    N2 = U_onsite * jnp.einsum("ok,kq,qk->koq", Vu, Ud, Vd, optimize="optimal")
    N3 = U_onsite * jnp.einsum("ko,kp,qk->okpq", Uu, Ud, Vd, optimize="optimal")
    N4 = U_onsite * jnp.einsum("ok,kq,pk->koqp", Vu, Ud, Vd, optimize="optimal")
    res += jnp.einsum("okp,koq,o,pq", N1, N2, t1u, t2d, optimize="optimal") * Du * Dd
    res -= jnp.einsum("okpq,koqp,o,pq", N3, N4, t1u, t2d, optimize="optimal") * Du * Dd
    res += U_onsite * U_onsite * jnp.einsum("io,oi,o,ip,pi,p", Uu, Vu, t1u, Ud, Vd, t1d, optimize="optimal") * Du * Dd
    return res


def E2eud_sq(X1uc, X2u, X1dc, X2d, w):
    # terms with Hw @ Hw
    if w is None:
        return 0

    Du, t1u, Uu, Vu = T(X1uc, X2u, 1)
    Dd, t1d, Ud, Vd = T(X1dc, X2d, 1)
    Du, t2u, Uu, Vu = T(X1uc, X2u, 2)
    Dd, t2d, Ud, Vd = T(X1dc, X2d, 2)

    resww = 0

    Wf = jnp.einsum("ikac,io,aq->okqc", w, Uu, Ud, optimize="optimal")
    Wg = jnp.einsum("jlcd,jp,qd->plcq", w, Uu, Vd, optimize="optimal")
    Wi = jnp.einsum("klbd,ol->kobd", w, Vu, optimize="optimal")

    # no contractions
    P = jnp.einsum("okqc,ok,qc->oq", Wf, Vu, Vd, optimize="optimal")
    Q = jnp.einsum("okqc,pk,qc->opq", Wf, Vu, Vd, optimize="optimal")
    R = jnp.einsum("okqc,ok,rc->oqr", Wf, Vu, Vd, optimize="optimal")
    S = jnp.einsum("okqc,pk,rc->opqr", Wf, Vu, Vd, optimize="optimal")
    resww += jnp.einsum("oq,pr,op,qr", P, P, t2u, t2d, optimize="optimal") * Du * Dd
    resww -= jnp.einsum("opq,por,op,qr", Q, Q, t2u, t2d, optimize="optimal") * Du * Dd
    resww -= jnp.einsum("oqr,prq,op,qr", R, R, t2u, t2d, optimize="optimal") * Du * Dd
    resww += jnp.einsum("opqr,porq,op,qr", S, S, t2u, t2d, optimize="optimal") * Du * Dd

    # one contraction of one  of them
    M1 = jnp.einsum("okqc,ok->oqc", Wf, Vu, optimize="optimal")
    M2 = jnp.einsum("plcq,pl->pcq", Wg, Vu, optimize="optimal")
    M3 = jnp.einsum("okqc,pk->opqc", Wf, Vu, optimize="optimal")
    M4 = jnp.einsum("plcq,ol->pocq", Wg, Vu, optimize="optimal")
    resww += jnp.einsum("oqc,pcq,op,q", M1, M2, t2u, t1d, optimize="optimal") * Du * Dd
    resww -= jnp.einsum("opqc,pocq,op,q", M3, M4, t2u, t1d, optimize="optimal") * Du * Dd

    N1 = jnp.einsum("okpc,pc->okp", Wf, Vd, optimize="optimal")
    N2 = jnp.einsum("kobd,bq,qd->koq", Wi, Ud, Vd, optimize="optimal")
    N3 = jnp.einsum("okpc,qc->okpq", Wf, Vd, optimize="optimal")
    N4 = jnp.einsum("kobd,bq,pd->koqp", Wi, Ud, Vd, optimize="optimal")
    resww += jnp.einsum("okp,koq,o,pq", N1, N2, t1u, t2d, optimize="optimal") * Du * Dd
    resww -= jnp.einsum("okpq,koqp,o,pq", N3, N4, t1u, t2d, optimize="optimal") * Du * Dd

    # one contraction each
    st = "ikac,klcd,io, ol,o, ap,pd, p"
    resww += jnp.einsum(st, w, w, Uu, Vu, t1u, Ud, Vd, t1d, optimize="optimal") * Du * Dd  # use symmetry of w
    return resww


def E1e_onsite_both(X1uc, X2u, X1dc, X2d, H1e, U_onsite):
    if H1e is None:
        return 0, 0
    Du, t1u, Uu, Vu = T(X1uc, X2u, 1)
    Dd, t1d, Ud, Vd = T(X1dc, X2d, 1)
    Du, t2u, Uu, Vu = T(X1uc, X2u, 2)
    Dd, t2d, Ud, Vd = T(X1dc, X2d, 2)
    res2w = 0
    for U_, V_, t2_, Uo, Vo, t1o in ((Uu, Vu, t2u, Ud, Vd, t1d), (Ud, Vd, t2d, Uu, Vu, t1u)):
        W0 = U_onsite * jnp.einsum("ko,kq,ok,qk->oq", U_, Uo, V_, Vo, optimize="optimal")
        W1 = U_onsite * jnp.einsum("ko,kq,pk,qk->opq", U_, Uo, V_, Vo, optimize="optimal")
        res2w += jnp.einsum("oq,p,op,q", W0, jnp.einsum("jl,jp,pl->p", H1e, U_, V_, optimize="optimal"), t2_, t1o, optimize="optimal") * Du * Dd
        res2w -= jnp.einsum("opq,po,op,q", W1, jnp.einsum("jl,jp,ol->po", H1e, U_, V_, optimize="optimal"), t2_, t1o, optimize="optimal") * Du * Dd

    resw2 = res2w.copy()
    res2w += U_onsite * jnp.einsum("li,io,ol,o,ip,pi,p", H1e, Uu, Vu, t1u, Ud, Vd, t1d, optimize="optimal") * Du * Dd
    res2w += U_onsite * jnp.einsum("li,io,ol,o,ip,pi,p", H1e, Ud, Vd, t1d, Uu, Vu, t1u, optimize="optimal") * Du * Dd
    resw2 += U_onsite * jnp.einsum("ik,io,ok,o,kp,pk,p", H1e, Uu, Vu, t1u, Ud, Vd, t1d, optimize="optimal") * Du * Dd
    resw2 += U_onsite * jnp.einsum("ik,io,ok,o,kp,pk,p", H1e, Ud, Vd, t1d, Uu, Vu, t1u, optimize="optimal") * Du * Dd
    return res2w, resw2


def expval_sq_spin(X1uc, X1dc, X2u, X2d, deriv_indices=None, hamiltonian=None, _sq=True):
    # compute <>, <H^2> and <H>
    # UHF version;
    # svd version; works with zero overlap

    old = ol(X1dc, X2d)
    olu = ol(X1uc, X2u)
    ol_ = olu * old

    if hamiltonian is None:
        return (ol_[None, None],) * (3 if _sq else 2)
    else:
        H2e = hamiltonian.get('H2e', None)
        H1e = hamiltonian.get('H1e', None)
        Enuc = hamiltonian.get('Enuc', 0)
        w = hamiltonian.get('w', None)
        U = hamiltonian.get('U', None)
        U_onsite = hamiltonian.get('U_onsite', None)
        if U is not None:
            raise NotImplementedError("U operators in the stable UHF <H^2>")
        if U_onsite is not None and (H2e is not None or w is not None):
            raise NotImplementedError("U_onsite cannot be combined with H2e or w")

    if H2e is not None:
        H2e, _, Htt = antisymmetrized(H2e)
    else:
        H2e = Htt = None

    Du, t1u, Uu, Vu = T(X1uc, X2u, 1)
    Dd, t1d, Ud, Vd = T(X1dc, X2d, 1)
    Du, t2u, Uu, Vu = T(X1uc, X2u, 2)
    Dd, t2d, Ud, Vd = T(X1dc, X2d, 2)
    _, t3u, _, _ = T(X1uc, X2u, 3)
    _, t3d, _, _ = T(X1dc, X2d, 3)

    eu1_ = E1e(X1uc, X2u, H1e)
    ed1_ = E1e(X1dc, X2d, H1e)
    eu2_ = E2e(X1uc, X2u, H2e)
    ed2_ = E2e(X1dc, X2d, H2e)

    eu1 = eu1_ * old
    ed1 = ed1_ * olu
    eu2 = eu2_ * old
    ed2 = ed2_ * olu

    if U_onsite is None:
        eud = E2eud(X1uc, X2u, X1dc, X2d, w)
    else:
        eud = Eonsiteud(X1uc, X2u, X1dc, X2d, U_onsite)

    expval = Enuc * old * olu + ed1 + ed2 + eu1 + eu2 + eud

    if not _sq:
        # match derivative interface for use in optimization
        return (old * olu)[None, None], expval[None, None]
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
    res22 += E1e_sq(X1uc, X2u, H1e) * old
    res22 += E1e_sq(X1dc, X2d, H1e) * olu
    # different spin
    res22 += 2 * eu1_ * ed1_  # use symmetry -> 2x

    # terms with H4
    res44 = 0
    # same spin
    res44 += E2e_sq(X1uc, X2u, H2e) * old
    res44 += E2e_sq(X1dc, X2d, H2e) * olu
    # different spin
    res44 += 2 * ed2_ * eu2_  # use symmetry -> 2x

    # terms with Hw @ Hw
    if U_onsite is None:
        resww = E2eud_sq(X1uc, X2u, X1dc, X2d, w)
    else:
        resww = Eonsiteud_sq(X1uc, X2u, X1dc, X2d, U_onsite)

    # terms with H4 @ H1e / H1e @ H4
    if H1e is not None and H2e is not None:
        different_spin = eu2_ * ed1_ + ed2_ * eu1_
        eu24, eu42 = E1e2e_both(X1uc, X2u, H1e, H2e)
        ed24, ed42 = E1e2e_both(X1dc, X2d, H1e, H2e)
        res42 = eu42 * old + ed42 * olu + different_spin
        res24 = eu24 * old + ed24 * olu + different_spin
    else:
        res42 = res24 = 0

    # terms with Hw @ H1e / H1e @ Hw
    if U_onsite is not None:
        res2w, resw2 = E1e_onsite_both(X1uc, X2u, X1dc, X2d, H1e, U_onsite)
    elif w is not None and H1e is not None:
        res2w = 0

        # no contractions

        for U_, V_, t2_, Uo, Vo, t1o in ((Uu, Vu, t2u, Ud, Vd, t1d), (Ud, Vd, t2d, Uu, Vu, t1u)):
            Wj = jnp.einsum("ikab,io,aq->okqb", w, U_, Uo, optimize="optimal")
            res2w += jnp.einsum("oq,p,op,q", jnp.einsum("okqb,ok,qb->oq", Wj, V_, Vo, optimize="optimal"), jnp.einsum("jl,jp,pl->p", H1e, U_, V_, optimize="optimal"), t2_, t1o, optimize="optimal") * Du * Dd
            res2w -= jnp.einsum("opq,po,op,q", jnp.einsum("okqb,pk,qb->opq", Wj, V_, Vo, optimize="optimal"), jnp.einsum("jl,jp,ol->po", H1e, U_, V_, optimize="optimal"), t2_, t1o, optimize="optimal") * Du * Dd

        # single contraction

        resw2 = res2w.copy()

        st = "ikab,kl,io,ol,o,ap,pb,p"
        res2w += jnp.einsum(st, w, H1e, Uu, Vu, t1u, Ud, Vd, t1d, optimize="optimal") * Du * Dd
        res2w += jnp.einsum(st, w, H1e, Ud, Vd, t1d, Uu, Vu, t1u, optimize="optimal") * Du * Dd

        st = "klab,ik,io,ol,o,ap,pb,p"
        resw2 += jnp.einsum(st, w, H1e, Uu, Vu, t1u, Ud, Vd, t1d, optimize="optimal") * Du * Dd
        resw2 += jnp.einsum(st, w, H1e, Ud, Vd, t1d, Uu, Vu, t1u, optimize="optimal") * Du * Dd
    else:
        res2w = resw2 = 0

    # terms with Hw @ H4 / H4 @ Hw
    if H2e is not None and w is not None:
        res4w = 0

        spins = (
            (Du, t1u, Uu, Vu, Dd, Ud, Vd, t2d, t3d),
            (Dd, t1d, Ud, Vd, Du, Uu, Vu, t2u, t3u),
        )

        sc4w = scw4 = 0
        for D_, t1_, U_, V_, Do, Uo, Vo, t2o, t3o in spins:
            Wt = jnp.einsum("ijad,io,oj->oad", w, U_, V_, optimize="optimal")
            Tht = jnp.einsum("bcef,bq,cr->qref", Htt, Uo, Uo, optimize="optimal")
            Thb = jnp.einsum("bcef,bq->qcef", Htt, Uo, optimize="optimal")

            # no contractions
            W1 = jnp.einsum("oad,ap,pd->op", Wt, Uo, Vo, optimize="optimal")
            W2 = jnp.einsum("oad,ap,qd->opq", Wt, Uo, Vo, optimize="optimal")
            H2 = jnp.einsum("qref,qe,rf->qr", Tht, Vo, Vo, optimize="optimal")
            H3 = jnp.einsum("qref,pe,rf->qpr", Tht, Vo, Vo, optimize="optimal")
            res4w -= 0.5 * jnp.einsum("op,qr,o,pqr", W1, H2, t1_, t3o, optimize="optimal") * D_ * Do
            res4w += jnp.einsum("opq,qpr,o,pqr", W2, H3, t1_, t3o, optimize="optimal") * D_ * Do

            # single contraction
            sc4w += jnp.einsum("opc,cqp,o,pq", jnp.einsum("oac,ap->opc", Wt, Uo, optimize="optimal"), jnp.einsum("qcef,pe,qf->cqp", Thb, Vo, Vo, optimize="optimal"), t1_, t2o, optimize="optimal") * D_ * Do
            scw4 -= jnp.einsum("oxp,xpq,o,pq", jnp.einsum("oxd,pd->oxp", Wt, Vo, optimize="optimal"), jnp.einsum("pqxf,qf->xpq", Tht, Vo, optimize="optimal"), t1_, t2o, optimize="optimal") * D_ * Do

        resw4 = res4w + scw4
        res4w = res4w + sc4w
    else:
        res4w = resw4 = 0

    expval_sq = 0
    expval_sq += res00 + 2 * res024 + 2 * res0w
    expval_sq += res22 + res24 + res42 + res2w + resw2
    expval_sq += res44 + res4w + resw4
    expval_sq += resww

    return (olu * old)[None, None], expval_sq[None, None], expval[None, None]


expval_spin = partial(expval_sq_spin, _sq=False)
