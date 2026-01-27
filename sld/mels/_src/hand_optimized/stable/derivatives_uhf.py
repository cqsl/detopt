import jax.numpy as jnp


from .common import A_inv_det_stable_1, A_inv_det_stable_2


def interaction_ud_deriv(Ucu, iu, Vu, ku, Ucd, kd, Vd, jd, w, mo_coeffu=None, mo_coeffd=None, swap=False):
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

    DMd, M_invd = A_inv_det_stable_1(Md)
    Dd, MMd = A_inv_det_stable_2(Md)

    DMu, M_invu = A_inv_det_stable_1(Mu)
    Du, MMu = A_inv_det_stable_2(Mu)

    r1u = jnp.einsum("pb,pq,qa->ab", Vu, M_invu, Ucu, optimize="optimal")
    r1d = jnp.einsum("re,rs,sg->eg", Vd, M_invd, Ucd, optimize="optimal")

    if mo_coeffu is None:
        assert mo_coeffd is None
        mo_coeff_arrays = ()
        if swap:
            abge = "geab"
        else:
            abge = "abge"
    else:
        assert mo_coeffd is not None
        mo_coeff_arrays = (mo_coeffu, mo_coeffu, mo_coeffd, mo_coeffd)
        if swap:
            abge = "GEAB,Aa,Bb,Gg,Ee"
        else:
            abge = "ABGE,Aa,Bb,Gg,Ee"

    # TODO ultimately rewrite this in terms of two 2x2 det / antisymmetrization
    res0 = res1 = res2 = res3 = res4 = 0

    res0 += detMu * detMd * jnp.einsum(f"{abge},b,g->ae", w, *mo_coeff_arrays, Vku, Uckd, optimize="optimal")
    res0 += detMu * DMd * jnp.einsum(f"{abge},b,j,eg->aj", w, *mo_coeff_arrays, Vku, Uckd, r1d, optimize="optimal")
    res0 += -detMu * DMd * jnp.einsum(f"{abge},b,g,ej->aj", w, *mo_coeff_arrays, Vku, Uckd, r1d, optimize="optimal")
    res0 += -detMu * DMd * jnp.einsum(f"{abge},b,y,yg->ae", w, *mo_coeff_arrays, Vku, Uckd, r1d, optimize="optimal")
    res0 += -detMu * Dd * jnp.einsum(f"{abge},b,wzmn,wy,y,zj,me,ng->aj", w, *mo_coeff_arrays, Vku, MMd, Vd, Uckd, Ucd, Vd, Ucd, optimize="branch-all")

    res1 += DMu * detMd * jnp.einsum(f"{abge},i,ab,g->ie", w, *mo_coeff_arrays, Vku, r1u, Uckd, optimize="optimal")
    res1 += DMu * DMd * jnp.einsum(f"{abge},i,ab,j,eg->ij", w, *mo_coeff_arrays, Vku, r1u, Uckd, r1d, optimize="optimal")
    res1 += -DMu * DMd * jnp.einsum(f"{abge},i,ab,g,ej->ij", w, *mo_coeff_arrays, Vku, r1u, Uckd, r1d, optimize="optimal")
    res1 += -DMu * DMd * jnp.einsum(f"{abge},i,ab,y,yg->ie", w, *mo_coeff_arrays, Vku, r1u, Uckd, r1d, optimize="optimal")
    res1 += -DMu * Dd * jnp.einsum(f"{abge},i,ab,wzmn,wy,y,zj,me,ng->ij", w, *mo_coeff_arrays, Vku, r1u, MMd, Vd, Uckd, Ucd, Vd, Ucd, optimize="branch-all")

    res2 += -DMu * detMd * jnp.einsum(f"{abge},xb,x,g->ae", w, *mo_coeff_arrays, r1u, Vku, Uckd, optimize="optimal")
    res2 += -DMu * DMd * jnp.einsum(f"{abge},xb,x,j,eg->aj", w, *mo_coeff_arrays, r1u, Vku, Uckd, r1d, optimize="optimal")
    res2 += DMu * DMd * jnp.einsum(f"{abge},xb,x,g,ej->aj", w, *mo_coeff_arrays, r1u, Vku, Uckd, r1d, optimize="optimal")
    res2 += DMu * DMd * jnp.einsum(f"{abge},xb,x,y,yg->ae", w, *mo_coeff_arrays, r1u, Vku, Uckd, r1d, optimize="optimal")
    res2 += DMu * Dd * jnp.einsum(f"{abge},xb,x,wzmn,wy,y,zj,me,ng->aj", w, *mo_coeff_arrays, r1u, Vku, MMd, Vd, Uckd, Ucd, Vd, Ucd, optimize="branch-all")

    res3 += -DMu * detMd * jnp.einsum(f"{abge},ai,b,g->ie", w, *mo_coeff_arrays, r1u, Vku, Uckd, optimize="optimal")
    res3 += -DMu * DMd * jnp.einsum(f"{abge},ai,b,j,eg->ij", w, *mo_coeff_arrays, r1u, Vku, Uckd, r1d, optimize="optimal")
    res3 += DMu * DMd * jnp.einsum(f"{abge},ai,b,g,ej->ij", w, *mo_coeff_arrays, r1u, Vku, Uckd, r1d, optimize="optimal")
    res3 += DMu * DMd * jnp.einsum(f"{abge},ai,b,y,yg->ie", w, *mo_coeff_arrays, r1u, Vku, Uckd, r1d, optimize="optimal")
    res3 += DMu * Dd * jnp.einsum(f"{abge},ai,b,wzmn,wy,y,zj,me,ng->ij", w, *mo_coeff_arrays, r1u, Vku, MMd, Vd, Uckd, Ucd, Vd, Ucd, optimize="branch-all")

    res4 += -Du * detMd * jnp.einsum(f"{abge},pqrs,pi,sa,rb,qx,x,g->ie", w, *mo_coeff_arrays, MMu, Vu, Ucu, Vu, Ucu, Vku, Uckd, optimize="branch-all")
    res4 += -Du * DMd * jnp.einsum(f"{abge},pqrs,pi,sa,rb,qx,x,j,eg->ij", w, *mo_coeff_arrays, MMu, Vu, Ucu, Vu, Ucu, Vku, Uckd, r1d, optimize="branch-all")
    res4 += Du * DMd * jnp.einsum(f"{abge},pqrs,pi,sa,rb,qx,x,g,ej->ij", w, *mo_coeff_arrays, MMu, Vu, Ucu, Vu, Ucu, Vku, Uckd, r1d, optimize="branch-all")
    res4 += Du * DMd * jnp.einsum(f"{abge},pqrs,pi,sa,rb,qx,x,y,yg->ie", w, *mo_coeff_arrays, MMu, Vu, Ucu, Vu, Ucu, Vku, Uckd, r1d, optimize="branch-all")

    # opt einsum takes a long time to optimize this, so we pre-contract some stuff by hand
    # res4 +=  Du  *Dd*  jnp.einsum(abge+'pqrs,pi,sa,rb,qx,x,wzmn,wy,y,zj,me,ng->ij', w, *mo_coeff_arrays, MMu, Vu, Ucu, Vu, Ucu, Vku,  MMd,  Vd, Uckd, Ucd,Vd, Ucd, optimize='branch-2')
    tmp = jnp.einsum(f"{abge},sa,rb,me,ng->srmn", w, *mo_coeff_arrays, Ucu, Vu, Vd, Ucd, optimize="optimal")
    res4 += Du * Dd * jnp.einsum("srmn,pqrs,pi,qx,x,wzmn,wy,y,zj->ij", tmp, MMu, Vu, Ucu, Vku, MMd, Vd, Uckd, Ucd, optimize="branch-all")

    res = res0 + res1 + res2 + res3 + res4

    assert jnp.issubdtype(ku.dtype, jnp.signedinteger)
    sign = 1 - 2 * ((iu + ku + kd + jd) % 2)  # (-1) ** (iu + ku + kd + jd)
    return sign * res
