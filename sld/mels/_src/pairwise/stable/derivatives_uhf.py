import jax.numpy as jnp
from string import ascii_letters

from .common import A_inv_det_stable_1, T
from sld.utils.antisymm import Sn


def _cofactor2(A):
    X, singular_values, YH = jnp.linalg.svd(A)
    n = A.shape[0]
    diagonal = jnp.eye(n, dtype=bool)
    mask = diagonal[:, :, None] | diagonal[:, None, :]
    products = (mask + (~mask) * singular_values[:, None, None]).prod(axis=0)
    determinant = jnp.linalg.det(X) * jnp.linalg.det(YH)
    cofactor = jnp.einsum("bc,jb,lc,bi,ck->ijkl", (~diagonal) * products, X, X, YH, YH, optimize="optimal")
    cofactor -= jnp.einsum("bc,lb,jc,bi,ck->ijkl", (~diagonal) * products, X, X, YH, YH, optimize="optimal")
    return determinant, cofactor.conj()


def _d_uc_rdm(Uc, i, V, k):
    Vk = V[k]
    V = jnp.delete(V, k, axis=0, assume_unique_indices=True)
    Uc = jnp.delete(Uc, i, axis=0, assume_unique_indices=True)
    M = Uc @ V.T
    identity = jnp.eye(V.shape[1])
    UcVk = Uc @ Vk

    result = -jnp.linalg.det(M) * jnp.einsum("ag,b->abg", identity, Vk, optimize="optimal")
    determinant, inverse = A_inv_det_stable_1(M)
    result -= determinant * jnp.einsum("g,ib,ij,ja->abg", Vk, V, inverse, Uc, optimize="optimal")
    result += determinant * jnp.einsum("ag,ib,ij,j->abg", identity, V, inverse, UcVk, optimize="optimal")
    result += determinant * jnp.einsum("ig,ij,ja,b->abg", V, inverse, Uc, Vk, optimize="optimal")
    determinant2, cofactor2 = _cofactor2(M)
    result += determinant2 * jnp.einsum("ijkl,ig,la,kb,j->abg", cofactor2, V, Uc, V, UcVk, optimize="optimal")
    sign = 1 - 2 * ((i + k) % 2)
    return -sign * result


def _d_v_rdm(Uc, k, V, j):
    Uck = Uc[k]
    V = jnp.delete(V, j, axis=0, assume_unique_indices=True)
    Uc = jnp.delete(Uc, k, axis=0, assume_unique_indices=True)
    M = Uc @ V.T
    identity = jnp.eye(V.shape[1])
    VUck = V @ Uck

    result = -jnp.linalg.det(M) * jnp.einsum("a,be->abe", Uck, identity, optimize="optimal")
    determinant, inverse = A_inv_det_stable_1(M)
    result -= determinant * jnp.einsum("e,kb,kl,la->abe", Uck, V, inverse, Uc, optimize="optimal")
    result += determinant * jnp.einsum("a,kb,kl,le->abe", Uck, V, inverse, Uc, optimize="optimal")
    result += determinant * jnp.einsum("i,ij,ja,be->abe", VUck, inverse, Uc, identity, optimize="optimal")
    determinant2, cofactor2 = _cofactor2(M)
    result += determinant2 * jnp.einsum("ijkl,i,je,kb,la->abe", cofactor2, VUck, Uc, V, Uc, optimize="optimal")
    sign = 1 - 2 * ((k + j) % 2)
    return -sign * result


def _einsum2(subscripts, ijkl, *args, D, t2, X, YH):
    available_letters = [x for x in ascii_letters if x not in subscripts]
    p, q, *_ = available_letters
    i, j, k, l = ijkl
    in_str, out_str = subscripts.split("->")

    res = 0
    for sgn, s in Sn(p + q):
        res += sgn * jnp.einsum(
            f"{in_str},{p}{i},{j}{s[0]},{q}{k},{l}{s[1]},{p}{q}->{out_str}",
            *args,
            YH.conj(),
            X.conj(),
            YH.conj(),
            X.conj(),
            t2,
            optimize="branch-all",
        )
    return D * res


def interaction_ud_deriv(Ucu, iu, Vu, ku, Ucd, kd, Vd, jd, w, swap=False, U_onsite=None):
    if U_onsite is not None:
        d_uc_u = _d_uc_rdm(Ucu, iu, Vu, ku)
        d_v_d = _d_v_rdm(Ucd, kd, Vd, jd)
        return U_onsite * jnp.einsum("aai,aaj->ij", d_uc_u, d_v_d, optimize="optimal")
    if w is None:
        return 0

    Vku = Vu[ku]
    Vu = jnp.delete(Vu, ku, axis=0, assume_unique_indices=True)
    Ucu = jnp.delete(Ucu, iu, axis=0, assume_unique_indices=True)
    Mu = Ucu @ Vu.T

    Uckd = Ucd[kd]
    Vd = jnp.delete(Vd, jd, axis=0, assume_unique_indices=True)
    Ucd = jnp.delete(Ucd, kd, axis=0, assume_unique_indices=True)
    Md = Ucd @ Vd.T

    detMu = jnp.linalg.det(Mu)
    detMd = jnp.linalg.det(Md)

    Dd, t2d, _, _, Xd, YHd = T(Ucd, Vd, 2, return_svd=True)
    Du, t2u, _, _, Xu, YHu = T(Ucu, Vu, 2, return_svd=True)

    _, t1d, Ud1, Wd1 = T(Ucd, Vd, 1)
    _, t1u, Uu1, Wu1 = T(Ucu, Vu, 1)
    r1u = jnp.einsum("bo,oa,o->ab", Uu1, Wu1, t1u, optimize="optimal")
    r1d = jnp.einsum("eo,og,o->eg", Ud1, Wd1, t1d, optimize="optimal")

    w = w if not swap else jnp.transpose(w, (2, 3, 0, 1))

    w_bra_g = jnp.einsum("abge,g->abe", w, Uckd, optimize="optimal")
    w_bra_ag = jnp.einsum("abe,sa->sbe", w_bra_g, Ucu, optimize="optimal")

    w_bra_Gfull = jnp.einsum("abge,ng->aben", w, Ucd, optimize="optimal")
    w_bra_AGfull = jnp.einsum("aben,sa->sben", w_bra_Gfull, Ucu, optimize="optimal")

    # (Uckd @ r1d)[g] = sum_o t1d[o] * (Uckd . Ud1[:,o]) * conj(Xd[n,o]) * Ucd[n,g]
    cy = jnp.einsum("y,yo->o", Uckd, Ud1, optimize="optimal")
    combined_n = jnp.einsum("o,no,o->n", t1d, Xd.conj(), cy, optimize="optimal")
    term4_reduced = jnp.einsum("aben,n->abe", w_bra_Gfull, combined_n, optimize="optimal")
    Xd_t1d = Xd.conj() * t1d[None, :]

    res0 = res1 = res2 = res3 = res4 = 0

    vw = Vd @ Uckd
    vq = Ucu @ Vku

    res0 += detMu * detMd * jnp.einsum("abe,b->ae", w_bra_g, Vku, optimize="optimal")
    res0 += detMu * Dd * jnp.einsum("abge,b,j,eg->aj", w, Vku, Uckd, r1d, optimize="optimal")
    res0 += -detMu * Dd * jnp.einsum("abe,b,ej->aj", w_bra_g, Vku, r1d, optimize="optimal")
    res0 += -detMu * Dd * jnp.einsum("abe,b->ae", term4_reduced, Vku, optimize="optimal")
    tmp0 = jnp.einsum("aben,b,me->amn", w_bra_Gfull, Vku, Vd, optimize="optimal")
    res0 += -detMu * _einsum2("amn,w,zj->aj", "wzmn", tmp0, vw, Ucd, D=Dd, t2=t2d, X=Xd, YH=YHd)

    res1 += Du * detMd * Vku[:, None] * jnp.einsum("abe,ab->e", w_bra_g, r1u, optimize="optimal")[None, :]
    res1 += Du * Dd * jnp.einsum("abge,i,ab,j,eg->ij", w, Vku, r1u, Uckd, r1d, optimize="optimal")
    res1 += -Du * Dd * Vku[:, None] * jnp.einsum("abe,ab,ej->j", w_bra_g, r1u, r1d, optimize="optimal")[None, :]
    res1 += -Du * Dd * Vku[:, None] * jnp.einsum("abe,ab->e", term4_reduced, r1u, optimize="optimal")[None, :]
    tmp1 = jnp.einsum("aben,ab,me->mn", w_bra_Gfull, r1u, Vd, optimize="optimal")
    res1 += -Du * jnp.einsum("i,j->ij", Vku, _einsum2("mn,w,zj->j", "wzmn", tmp1, vw, Ucd, D=Dd, t2=t2d, X=Xd, YH=YHd))

    res2 += -Du * detMd * jnp.einsum("abe,xb,x->ae", w_bra_g, r1u, Vku, optimize="optimal")
    res2 += -Du * Dd * jnp.einsum("abge,xb,x,j,eg->aj", w, r1u, Vku, Uckd, r1d, optimize="optimal")
    res2 += Du * Dd * jnp.einsum("abe,xb,x,ej->aj", w_bra_g, r1u, Vku, r1d, optimize="optimal")
    res2 += Du * Dd * jnp.einsum("abe,xb,x->ae", term4_reduced, r1u, Vku, optimize="optimal")
    tmp2 = jnp.einsum("aben,xb,x,me->amn", w_bra_Gfull, r1u, Vku, Vd, optimize="optimal")
    res2 += Du * _einsum2("amn,w,zj->aj", "wzmn", tmp2, vw, Ucd, D=Dd, t2=t2d, X=Xd, YH=YHd)

    res3 += -Du * detMd * jnp.einsum("abe,ai,b->ie", w_bra_g, r1u, Vku, optimize="optimal")
    res3 += -Du * Dd * jnp.einsum("abge,ai,b,j,eg->ij", w, r1u, Vku, Uckd, r1d, optimize="optimal")
    res3 += Du * Dd * jnp.einsum("abe,ai,b,ej->ij", w_bra_g, r1u, Vku, r1d, optimize="optimal")
    res3 += Du * Dd * jnp.einsum("abe,ai,b->ie", term4_reduced, r1u, Vku, optimize="optimal")
    tmp3 = jnp.einsum("aben,ai,b,me->imn", w_bra_Gfull, r1u, Vku, Vd, optimize="optimal")
    res3 += Du * _einsum2("imn,w,zj->ij", "wzmn", tmp3, vw, Ucd, D=Dd, t2=t2d, X=Xd, YH=YHd)

    tmp4a = jnp.einsum("sbe,rb->sre", w_bra_ag, Vu, optimize="optimal")
    w_bra_e_ket = jnp.einsum("aben,eo->abno", w_bra_Gfull, Ud1, optimize="optimal")
    tmp4b_reduced = jnp.einsum("abno,no->ab", w_bra_e_ket, Xd_t1d, optimize="optimal")
    tmp4b = jnp.einsum("ab,sa,rb->sr", tmp4b_reduced, Ucu, Vu, optimize="optimal")
    tmp4c = jnp.einsum("sbe,rb,ej->srj", w_bra_ag, Vu, r1d, optimize="optimal")
    tmp4d = jnp.einsum("abe,sa,rb->sre", term4_reduced, Ucu, Vu, optimize="optimal")
    res4 += -detMd * _einsum2("sre,pi,q->ie", "pqrs", tmp4a, Vu, vq, D=Du, t2=t2u, X=Xu, YH=YHu)
    res4 += -Dd * jnp.einsum("i,j->ij", _einsum2("sr,pi,q->i", "pqrs", tmp4b, Vu, vq, D=Du, t2=t2u, X=Xu, YH=YHu), Uckd)
    res4 += Dd * _einsum2("srj,pi,q->ij", "pqrs", tmp4c, Vu, vq, D=Du, t2=t2u, X=Xu, YH=YHu)
    res4 += Dd * _einsum2("sre,pi,q->ie", "pqrs", tmp4d, Vu, vq, D=Du, t2=t2u, X=Xu, YH=YHu)

    tmp = jnp.einsum("sben,rb,me->srmn", w_bra_AGfull, Vu, Vd, optimize="optimal")
    tmp = _einsum2("srmn,w,zj->srj", "wzmn", tmp, vw, Ucd, D=Dd, t2=t2d, X=Xd, YH=YHd)
    res4 += _einsum2("srj,pi,q->ij", "pqrs", tmp, Vu, vq, D=Du, t2=t2u, X=Xu, YH=YHu)

    res = res0 + res1 + res2 + res3 + res4

    assert jnp.issubdtype(ku.dtype, jnp.signedinteger)
    sign = 1 - 2 * ((iu + ku + kd + jd) % 2)  # (-1) ** (iu + ku + kd + jd)
    return sign * res
