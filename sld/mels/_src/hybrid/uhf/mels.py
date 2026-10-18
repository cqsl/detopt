import jax.numpy as jnp
from sld.utils.antisymm import Sn
from ..algebra import constant_like, contract, result_type, coeff_at, trace
from ..common import _factors, _two_body_energy, inverse_polynomial


def H_A(Aconj, Htt_ea):
    return None if Htt_ea is None else contract('aebg,ja->jebg', Htt_ea, Aconj)


def _sector_energy(Aconj, B, Htt_bra, H1e, rcond, max_degree, factorization, AS=None):
    dtype = result_type(Aconj, B, 0 if Htt_bra is None else Htt_bra, 0 if H1e is None else H1e)
    P, det, null = inverse_polynomial(Aconj, B, rcond, max_degree, factorization, AS)
    L = B.T @ P
    K = None if Htt_bra is None else contract('ek,kebg->bg', L, Htt_bra)
    energy = constant_like(jnp.zeros((), dtype=dtype), L)
    if K is not None:
        energy = energy + _two_body_energy(Aconj, L, K)
    if H1e is not None:
        energy = energy + trace(Aconj @ H1e @ L)
    return L, K, det, null, energy, dtype


def _cross_contractions(L, Lo, wAo, wA):
    active_operator = contract('eq,ebaq->ab', Lo, wAo)
    other_operator = None if wA is None else contract('bq,qegb->ge', L, wA)
    return active_operator, other_operator


def _total_energy(A, L, ea, eo, cross, Enuc):
    energy = ea + eo + Enuc
    if cross is not None:
        energy = energy + trace(A @ cross @ L)
    return energy


def _rotation_overlap(Au, Ad, Bu, Bd, U):
    Uu, Ud = U
    return jnp.linalg.det(Au @ Uu @ Bu.T) * jnp.linalg.det(Ad @ Ud @ Bd.T)


def _expval_pair(Au, Ad, Bu, Bd, HAu, HAd, wAo, H1u, H1d, Enuc, rcond, max_degree, factorization, U=None, ASu=None, ASd=None):
    L_up, _, detu, nullu, eu, dtype = _sector_energy(Au, Bu, HAu, H1u, rcond, max_degree, factorization, ASu)
    L_down, _, detd, nulld, ed, _ = _sector_energy(Ad, Bd, HAd, H1d, rcond, max_degree, factorization, ASd)
    cross = None
    if wAo is not None:
        cross, _ = _cross_contractions(L_up, L_down, wAo, None)
    energy = _total_energy(Au, L_up, eu, ed, cross, Enuc)
    det = detu * detd
    null = None if nullu is None or nulld is None else nullu + nulld
    value = det * coeff_at(energy, null)
    if U is not None:
        value = value + _rotation_overlap(Au, Ad, Bu, Bd, U)
    overlap = det if null is None else det * coeff_at(constant_like(jnp.ones((), dtype=dtype), L_up), null)
    return overlap, value


def _expval_sq_pair(Au, Ad, Bu, Bd, H1u, H1d, Htt, w, Enuc, rcond, max_degree, factorization, sq_up=(None, None, None), sq_down=(None, None, None), ASu=None, ASd=None):
    Sinv, SiH_up, HSi_up = sq_up
    _, SiH_down, HSi_down = sq_down
    dtype = result_type(Au, Ad, Bu, Bd, *(0 if part is None else part for part in (H1u, H1d, Htt, w)))
    _, Lu, detu, nullu = _factors(Au, Bu, rcond, max_degree, factorization, ASu)
    _, Ld, detd, nulld = _factors(Ad, Bd, rcond, max_degree, factorization, ASd)
    Ru, Rd = Lu @ Au, Ld @ Ad
    zero = constant_like(jnp.zeros((), dtype=dtype), Lu)
    e1u = zero if H1u is None else contract('ba,ab', Ru, H1u)
    e1d = zero if H1d is None else contract('ba,ab', Rd, H1d)
    e2u = zero if Htt is None else 0.5 * contract('abge,ea,gb', Htt, Ru, Ru)
    e2d = zero if Htt is None else 0.5 * contract('abge,ea,gb', Htt, Rd, Rd)
    eud = zero if w is None else contract('ijkl,ij,kl', w, Rd, Ru)
    energy = jnp.asarray(Enuc, dtype=dtype) + e1u + e1d + e2u + e2d + eud
    terms = [jnp.asarray(Enuc * Enuc, dtype=dtype), 2 * Enuc * (e1u + e2u + e1d + e2d), 2 * Enuc * eud]
    for R, H, SiH in ((Ru, H1u, SiH_up), (Rd, H1d, SiH_down)):
        if H is None:
            continue
        terms.append(contract('ix,xj,ij', H, H if SiH is None else SiH, R))
        for sign, order in Sn('kl'):
            terms.append(sign * contract(f'i{order[0]},j{order[1]},ik,jl', H, H, R, R))
    if H1u is not None and H1d is not None:
        terms.append(2 * contract('ij,kl,ij,kl', H1d, H1u, Rd, Ru))
    if Htt is not None:
        for R, L, Aconj in ((Ru, Lu, Au), (Rd, Ld, Ad)):
            G4 = contract('ijkl,io,jp,qk,rl->opqr', Htt, L, L, Aconj, Aconj)
            diagonal = contract('opop->', G4)
            terms.append(0.25 * diagonal * diagonal)
            terms.append(0.25 * contract('opqr,qrop', G4, G4))
            terms.append(-contract('ijkc,abld,ik,jl,ac,bd', Htt, Htt, R, R, R, R))
            if Sinv is None:
                terms.append(contract('ijxu,ukyz,ix,jy,kz', Htt, Htt, R, R, R))
            else:
                terms.append(contract('ijxu,uv,vkyz,ix,jy,kz', Htt, Sinv, Htt, R, R, R))
            Cx = 0.5 * contract('ijxu,io,jp,qx->opqu', Htt, L, L, Aconj)
            Dx = contract('ukyz,kq,oy,pz->uqop', Htt, L, Aconj, Aconj)
            if Sinv is None:
                terms.append(contract('opqu,uqop', Cx, Dx))
            else:
                terms.append(contract('opqu,uv,vqop', Cx, Sinv, Dx))
            left_pair = contract('ijab,io,jp->opab', Htt, L, L)
            right_pair = contract('abkl,ok,pl->abop', Htt, Aconj, Aconj)
            if Sinv is None:
                terms.append(0.25 * contract('opab,abop', left_pair, right_pair))
            else:
                terms.append(0.25 * contract('opab,ax,by,xyop', left_pair, Sinv, Sinv, right_pair))
        terms.append(2 * e2d * e2u)
    if w is not None:
        for sign_u, order_u in Sn('kl'):
            for sign_d, order_d in Sn('cd'):
                subscripts = f'ikac,jlbd,i{order_u[0]},j{order_u[1]},a{order_d[0]},b{order_d[1]}'
                terms.append(sign_u * sign_d * contract(subscripts, w, w, Ru, Ru, Rd, Rd))
        for sign, order in Sn('kl'):
            if Sinv is None:
                value = contract(f'ikac,jlcd,i{order[0]},j{order[1]},ad', w, w, Ru, Ru, Rd)
            else:
                value = contract(f'ikac,cx,jlxd,i{order[0]},j{order[1]},ad', w, Sinv, w, Ru, Ru, Rd)
            terms.append(sign * value)
        for sign, order in Sn('cd'):
            if Sinv is None:
                value = contract(f'ikac,klbd,il,a{order[0]},b{order[1]}', w, w, Ru, Rd, Rd)
            else:
                value = contract(f'ikac,kx,xlbd,il,a{order[0]},b{order[1]}', w, Sinv, w, Ru, Rd, Rd)
            terms.append(sign * value)
        A = contract('ikac,io,ap->okpc', w, Lu, Ld)
        B = contract('klcd,ol,pd->kocp', w, Au, Ad)
        if Sinv is None:
            terms.append(contract('okpc,kocp', A, B))
        else:
            terms.append(contract('okpc,kx,cy,xoyp', A, Sinv, Sinv, B))
    if Htt is not None and (H1u is not None or H1d is not None):
        ordered = []
        for R, H in ((Ru, H1u), (Rd, H1d)):
            if H is None:
                continue
            ordered.append(-0.5 * contract('ijkl,ac,ik,jl,ac', Htt, H, R, R, R))
            ordered.append(contract('ijkc,al,ik,jl,ac', Htt, H, R, R, R))
        if H1d is not None:
            ordered.append(e2u * e1d)
        if H1u is not None:
            ordered.append(e2d * e1u)
        terms.append(2 * sum(ordered, start=zero))
        for R, H, SiH, HSi in ((Ru, H1u, SiH_up, HSi_up), (Rd, H1d, SiH_down, HSi_down)):
            if H is None:
                continue
            terms.append(-contract('ijxu,uy,xi,yj', Htt, H if SiH is None else SiH, R, R))
            terms.append(-contract('xuij,yu,ix,jy', Htt, H if HSi is None else HSi, R, R))
    if w is not None and (H1u is not None or H1d is not None):
        ordered = []
        for R, Ro, H in ((Ru, Rd, H1u), (Rd, Ru, H1d)):
            if H is None:
                continue
            for sign, order in Sn('kl'):
                ordered.append(sign * contract(f'ikab,jl,i{order[0]},j{order[1]},ab', w, H, R, R, Ro))
        terms.append(2 * sum(ordered, start=zero))
        for R, Ro, H, SiH, HSi in ((Ru, Rd, H1u, SiH_up, HSi_up), (Rd, Ru, H1d, SiH_down, HSi_down)):
            if H is None:
                continue
            terms.append(contract('ikab,kl,il,ab', w, H if SiH is None else SiH, R, Ro))
            terms.append(contract('klab,ik,il,ab', w, H if HSi is None else HSi, R, Ro))
    if Htt is not None and w is not None:
        ordered = []
        for R, Ro in ((Ru, Rd), (Rd, Ru)):
            ordered.append(-0.5 * contract('ijad,bcef,ij,ad,be,cf', w, Htt, R, Ro, Ro, Ro))
            ordered.append(contract('ijad,bcef,ij,ae,bd,cf', w, Htt, R, Ro, Ro, Ro))
        terms.append(2 * sum(ordered, start=zero))
        for R, Ro in ((Ru, Rd), (Rd, Ru)):
            if Sinv is None:
                terms.append(contract('ijac,bcef,ij,ae,bf', w, Htt, R, Ro, Ro))
                terms.append(-contract('ijxd,bcxf,ij,bd,cf', w, Htt, R, Ro, Ro))
            else:
                terms.append(contract('ijac,cy,byef,ij,ae,bf', w, Sinv, Htt, R, Ro, Ro))
                terms.append(-contract('ijxd,xy,bcyf,ij,bd,cf', w, Sinv, Htt, R, Ro, Ro))
    squared = sum(terms, start=zero)
    det = detu * detd
    null = None if nullu is None or nulld is None else nullu + nulld
    overlap = det if null is None else det * coeff_at(constant_like(jnp.ones((), dtype=dtype), Lu), null)
    return overlap, det * coeff_at(squared, null), det * coeff_at(energy, null)


def _prepare_Htt_sector(Aconj, Htt, with_one_body):
    V = contract('ijcd,md->ijcm', Htt, Aconj)
    Y = contract('ijkr,qk->ijqr', V, Aconj)
    UA = contract('abge,ma->mbge', Htt, Aconj)
    W = contract('mbge,nb->mnge', UA, Aconj)
    U = jnp.moveaxis(UA, 2, 1) if with_one_body else None
    Z = contract('ijab,abop->ijop', Htt, Y)
    return V, Y, U, W, Z


def _prepare_w_pair(w, A_second, A_fourth, with_Z):
    V2 = contract('ikac,qk->iqac', w, A_second)
    V4 = contract('ikac,rc->ikar', w, A_fourth)
    Y = contract('iqac,rc->iqar', V2, A_fourth)
    Z = contract('ikac,kocp->iaop', w, Y) if with_Z else None
    return V2, V4, Y, Z


def _expval_sq_prepare_A(Au, Ad, Htt, w, with_one_body):
    sectors = None if Htt is None else tuple(_prepare_Htt_sector(Aconj, Htt, with_one_body) for Aconj in (Au, Ad))
    if w is None:
        return sectors, None
    opposite = (
        _prepare_w_pair(w, Au, Ad, True),
        _prepare_w_pair(w, Ad, Au, False),
    )
    return sectors, opposite


def _sq_sector_terms(prepared, L, R, H1e, terms):
    V, Y, U, W, Z = prepared
    e2 = 0.5 * contract('mnge,em,gn', W, L, L)
    G4 = contract('ijqr,io,jp->qrop', Y, L, L)
    diagonal = contract('ijop,io,jp->', Y, L, L)
    terms.append(0.25 * diagonal * diagonal)
    terms.append(0.25 * contract('opqr,qrop', G4, G4))
    half = -contract('ijcm,im->jc', V, L)
    terms.append(-contract('jc,al,jl,ac', half, half, R, R))
    terms.append(contract('ju,uy,jy', half, half, R))
    Cx = contract('juqo,jp->uqop', contract('ijuq,io->juqo', V, L), L)
    Dx = contract('ukop,kq->uopq', Y, L)
    terms.append(-0.5 * contract('uqop,uopq', Cx, Dx))
    terms.append(0.25 * contract('ijop,io,jp', Z, L, L))
    if H1e is None:
        return e2, diagonal, half, None
    return e2, diagonal, half, contract('mxju,xm->ju', U, L)


def _expval_sq_pair_prepared(Au, Ad, Bu, Bd, H1u, H1d, prepared, Enuc, dtype, rcond, max_degree, factorization, ASu=None, ASd=None):
    sectors, opposite = (None, None) if prepared is None else prepared
    _, Lu, detu, nullu = _factors(Au, Bu, rcond, max_degree, factorization, ASu)
    _, Ld, detd, nulld = _factors(Ad, Bd, rcond, max_degree, factorization, ASd)
    Ru, Rd = Lu @ Au, Ld @ Ad
    zero = constant_like(jnp.zeros((), dtype=dtype), Lu)
    e1u = zero if H1u is None else contract('ba,ab', Ru, H1u)
    e1d = zero if H1d is None else contract('ba,ab', Rd, H1d)

    same_spin = []
    if sectors is None:
        e2u = e2d = zero
        diagonal_u = diagonal_d = half_u = half_d = cross_u = cross_d = None
    else:
        e2u, diagonal_u, half_u, cross_u = _sq_sector_terms(sectors[0], Lu, Ru, H1u, same_spin)
        e2d, diagonal_d, half_d, cross_d = _sq_sector_terms(sectors[1], Ld, Rd, H1d, same_spin)

    if opposite is None:
        eud = zero
        bundles = ()
    else:
        w_ud, w_du = opposite
        eud = contract('iqar,iq,ar', w_du[2], Ld, Lu)
        # (w tensors, this-spin factors, other-spin factors, this-spin H1, other-spin Htt cache)
        bundles = (
            (w_ud, Lu, Ru, Ld, Rd, H1u, None if sectors is None else sectors[1], diagonal_d),
            (w_du, Ld, Rd, Lu, Ru, H1d, None if sectors is None else sectors[0], diagonal_u),
        )

    energy = jnp.asarray(Enuc, dtype=dtype) + e1u + e1d + e2u + e2d + eud
    terms = [jnp.asarray(Enuc * Enuc, dtype=dtype), 2 * Enuc * (e1u + e2u + e1d + e2d), 2 * Enuc * eud]
    for R, H in ((Ru, H1u), (Rd, H1d)):
        if H is None:
            continue
        terms.append(contract('ix,xj,ij', H, H, R))
        for sign, order in Sn('kl'):
            terms.append(sign * contract(f'i{order[0]},j{order[1]},ik,jl', H, H, R, R))
    if H1u is not None and H1d is not None:
        terms.append(2 * contract('ij,kl,ij,kl', H1d, H1u, Rd, Ru))
    if sectors is not None:
        terms += same_spin
        terms.append(2 * e2d * e2u)
    if opposite is not None:
        V2u, V4d, Yud, Zw = w_ud
        for sign_u, order_u in Sn('qs'):
            for sign_d, order_d in Sn('rt'):
                subscripts = f'iqar,jsbt,i{order_u[0]},j{order_u[1]},a{order_d[0]},b{order_d[1]}'
                terms.append(sign_u * sign_d * contract(subscripts, Yud, Yud, Lu, Lu, Ld, Ld))
        for sign, order in Sn('qs'):
            terms.append(sign * contract(f'iqac,jsct,i{order[0]},j{order[1]},at', V2u, Yud, Lu, Lu, Ld))
        for sign, order in Sn('rt'):
            terms.append(sign * contract(f'ikar,ksbt,is,a{order[0]},b{order[1]}', V4d, Yud, Lu, Ld, Ld))
        terms.append(contract('iaop,io,ap', Zw, Lu, Ld))
    if sectors is not None and (H1u is not None or H1d is not None):
        ordered = []
        for half, R, H in ((half_u, Ru, H1u), (half_d, Rd, H1d)):
            if H is None:
                continue
            ordered.append(-0.5 * contract('jl,jl,ac,ac', half, R, H, R))
            ordered.append(contract('jc,jl,al,ac', half, R, H, R))
        if H1d is not None:
            ordered.append(e2u * e1d)
        if H1u is not None:
            ordered.append(e2d * e1u)
        terms.append(2 * sum(ordered, start=zero))
        for cross, R, H in ((cross_u, Ru, H1u), (cross_d, Rd, H1d)):
            if H is None:
                continue
            terms.append(-contract('ju,uy,yj', cross, H, R))
            terms.append(-contract('uj,yu,jy', cross, H, R))
    if opposite is not None and (H1u is not None or H1d is not None):
        ordered = []
        for w_pair, L, R, Lo, Ro, H, _, _ in bundles:
            if H is None:
                continue
            wY = w_pair[2]
            ordered.append(contract('iqar,jl,iq,jl,ar', wY, H, L, R, Lo))
            ordered.append(-contract('iqar,jl,il,jq,ar', wY, H, R, L, Lo))
        terms.append(2 * sum(ordered, start=zero))
        for w_pair, L, R, Lo, Ro, H, _, _ in bundles:
            if H is None:
                continue
            terms.append(contract('ikar,kl,il,ar', w_pair[1], H, R, Lo))
            terms.append(contract('kqar,ik,iq,ar', w_pair[2], H, L, Lo))
    if sectors is not None and opposite is not None:
        ordered = []
        for w_pair, L, R, Lo, Ro, _, other, other_diagonal in bundles:
            Yo = other[1]
            ordered.append(-0.5 * contract('iqar,iq,ar', w_pair[2], L, Lo) * other_diagonal)
            ordered.append(contract('iqar,bctu,iq,at,br,cu', w_pair[2], Yo, L, Lo, Lo, Lo))
        terms.append(2 * sum(ordered, start=zero))
        for w_pair, L, R, Lo, Ro, _, other, _ in bundles:
            Vo, Yo = (other[0], other[1])
            terms.append(contract('iqac,bctu,iq,at,bu', w_pair[0], Yo, L, Lo, Lo))
            terms.append(-contract('iqxr,bcxm,iq,br,cm', w_pair[2], Vo, L, Lo, Lo))
    squared = sum(terms, start=zero)
    det = detu * detd
    null = None if nullu is None or nulld is None else nullu + nulld
    overlap = det if null is None else det * coeff_at(constant_like(jnp.ones((), dtype=dtype), Lu), null)
    return overlap, det * coeff_at(squared, null), det * coeff_at(energy, null)
