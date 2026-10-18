from itertools import product

import jax.numpy as jnp
from sld.utils.antisymm import Sn
from ..keepspin import KeepSpinTwoBody, keepspin_assemble, keepspin_blocks, keepspin_einsum, orbital_free_letters
from ..algebra import constant_like, contract, result_type, coeff_at, trace, zeros_like
from ..common import _factors, _two_body_energy, _uses_enuc


def _expval_pair(Aconj, B, has_two_body, AH, Enuc, K_from, rcond, max_degree, factorization, AS=None):
    P, L, det, null = _factors(Aconj, B, rcond, max_degree, factorization, AS)
    one = constant_like(jnp.ones((), dtype=result_type(Aconj, B)), P)
    energy = zeros_like(one)
    if _uses_enuc(Enuc):
        energy = energy + Enuc
    if AH is not None:
        energy = energy + trace(AH @ L)
    if has_two_body:
        energy = energy + _two_body_energy(Aconj, L, K_from(P, L))
    overlap = det if null is None else det * coeff_at(one, null)
    return overlap, det * coeff_at(energy, null)


def _squared_one_body(R, H1e, contract, zero, SiH=None):
    if H1e is None:
        return zero, []
    terms = [contract('ix,xj,ij', H1e, H1e if SiH is None else SiH, R)]
    for sign, order in Sn('kl'):
        terms.append(sign * contract(f'i{order[0]},j{order[1]},ik,jl', H1e, H1e, R, R))
    return contract('ba,ab', R, H1e), terms


def _expval_sq_pair(Aconj, B, H1e, Htt, Enuc, rcond, max_degree, factorization, sq_metric=(None, None, None), AS=None):
    Sinv, SiH, HSi = sq_metric
    keepspin = isinstance(Htt, KeepSpinTwoBody)
    coefficients = Htt.arrays if keepspin else (Htt,)
    dtype = result_type(Aconj, B, *(0 if part is None else part for part in (H1e, *coefficients)))
    nmo = coefficients[0].shape[0] if keepspin else None
    _, L, det, null = _factors(Aconj, B, rcond, max_degree, factorization, AS)
    R = L @ Aconj
    zero = constant_like(jnp.zeros((), dtype=dtype), L)
    def contract_H2e(subscripts, spaces, *operands):
        if not keepspin:
            return contract(subscripts, *operands)
        return sum((sign * contract(subscripts, *args) for sign, args, _ in keepspin_einsum(subscripts, spaces, operands, nmo)), start=zero)

    def half(subscripts, spaces, *operands):
        if not keepspin:
            return contract(subscripts, *operands)
        return keepspin_assemble(subscripts, spaces, keepspin_blocks(subscripts, spaces, operands, nmo, contract))
    e1, one_body = _squared_one_body(R, H1e, contract, zero, SiH)
    e2 = zero if Htt is None else 0.5 * contract_H2e('abge,ea,gb', 'oooo,oo,oo', Htt, R, R)
    energy = jnp.asarray(Enuc, dtype=dtype) + e1 + e2
    terms = [jnp.asarray(Enuc * Enuc, dtype=dtype), 2 * Enuc * (e1 + e2)]
    terms += one_body
    if Htt is not None:
        G4 = contract_H2e('ijkl,io,jp,qk,rl->opqr', 'oooo,oe,oe,eo,eo', Htt, L, L, Aconj, Aconj)
        diagonal = contract('opop->', G4)
        terms.append(0.25 * diagonal * diagonal)
        terms.append(0.25 * contract('opqr,qrop', G4, G4))
        left = half('ijkc,ik->jc', 'oooo,oo', Htt, R)
        right = half('abld,bd->al', 'oooo,oo', Htt, R)
        terms.append(-contract('jc,al,jl,ac', left, right, R, R))
        left = half('ijxu,ix->ju', 'oooo,oo', Htt, R)
        right = half('ukyz,kz->uy', 'oooo,oo', Htt, R)
        if Sinv is None:
            terms.append(contract('ju,uy,jy', left, right, R))
        else:
            terms.append(contract('ju,uv,vy,jy', left, Sinv, right, R))
        Cx = half('ijxu,io,jp,qx->uopq', 'oooo,oe,oe,eo', Htt, L, L, Aconj)
        Dx = half('ukyz,kq,oy,pz->uqop', 'oooo,oe,eo,eo', Htt, L, Aconj, Aconj)
        if Sinv is None:
            terms.append(0.5 * contract('uopq,uqop', Cx, Dx))
        else:
            terms.append(0.5 * contract('uopq,uv,vqop', Cx, Sinv, Dx))
        left_pair = half('ijab,io,jp->opab', 'oooo,oe,oe', Htt, L, L)
        right_pair = half('abkl,ok,pl->abop', 'oooo,eo,eo', Htt, Aconj, Aconj)
        if Sinv is None:
            terms.append(0.25 * contract('opab,abop', left_pair, right_pair))
        else:
            terms.append(0.25 * contract('opab,ax,by,xyop', left_pair, Sinv, Sinv, right_pair))
    if Htt is not None and H1e is not None:
        ordered = [-0.5 * contract_H2e('ijkl,ac,ik,jl,ac', 'oooo,oo,oo,oo,oo', Htt, H1e, R, R, R), contract_H2e('ijkc,al,ik,jl,ac', 'oooo,oo,oo,oo,oo', Htt, H1e, R, R, R)]
        terms.append(2 * sum(ordered, start=zero))
        terms.append(-contract_H2e('ijxu,uy,xi,yj', 'oooo,oo,oo,oo', Htt, H1e if SiH is None else SiH, R, R))
        terms.append(-contract_H2e('xuij,yu,ix,jy', 'oooo,oo,oo,oo', Htt, H1e if HSi is None else HSi, R, R))
    squared = sum(terms, start=zero)
    one = constant_like(jnp.ones((), dtype=dtype), L)
    overlap = det if null is None else det * coeff_at(one, null)
    return overlap, det * coeff_at(squared, null), det * coeff_at(energy, null)


def _keepspin_dense(subscripts, spaces, operands, nmo):
    blocks = keepspin_blocks(subscripts, spaces, operands, nmo, contract)
    free = orbital_free_letters(subscripts, spaces)
    zero = zeros_like(next(iter(blocks.values())))
    for key in product((0, 1), repeat=len(free)):
        blocks.setdefault(key, zero)
    return keepspin_assemble(subscripts, spaces, blocks)


def _expval_sq_prepare_A(Aconj, Htt, with_one_body, Sinv=None):
    keepspin = isinstance(Htt, KeepSpinTwoBody)
    nmo = Htt.arrays[0].shape[0] if keepspin else None

    def two_body_contract(subscripts, spaces, *operands):
        if not keepspin:
            return contract(subscripts, *operands)
        return _keepspin_dense(subscripts, spaces, operands, nmo)
    V = two_body_contract('ijcd,md->ijcm', 'oooo,eo', Htt, Aconj)
    Y = contract('ijkr,qk->ijqr', V, Aconj)
    UA = two_body_contract('abge,ma->mbge', 'oooo,eo', Htt, Aconj)
    W = contract('mbge,nb->mnge', UA, Aconj)
    U = jnp.moveaxis(UA, 2, 1) if with_one_body else None
    if Sinv is None:
        Z = two_body_contract('ijab,abop->ijop', 'oooo,ooee', Htt, Y)
    else:
        Ydress = contract('ax,by,xyop->abop', Sinv, Sinv, Y)
        Z = two_body_contract('ijab,abop->ijop', 'oooo,ooee', Htt, Ydress)
    return V, Y, U, W, Z


def _expval_sq_pair_prepared(Aconj, B, H1e, prepared, Enuc, dtype, rcond, max_degree, factorization, sq_metric=(None, None, None), AS=None):
    Sinv, SiH, HSi = sq_metric
    V, Y, U, W, Z = (None,) * 5 if prepared is None else prepared
    _, L, det, null = _factors(Aconj, B, rcond, max_degree, factorization, AS)
    R = L @ Aconj
    zero = constant_like(jnp.zeros((), dtype=dtype), L)
    e1, one_body = _squared_one_body(R, H1e, contract, zero, SiH)
    e2 = zero if prepared is None else 0.5 * contract('mnge,em,gn', W, L, L)
    energy = jnp.asarray(Enuc, dtype=dtype) + e1 + e2
    terms = [jnp.asarray(Enuc * Enuc, dtype=dtype), 2 * Enuc * (e1 + e2)]
    terms += one_body
    if prepared is not None:
        G4 = contract('ijqr,io,jp->qrop', Y, L, L)
        trace = contract('ijop,io,jp->', Y, L, L)
        terms.append(0.25 * trace * trace)
        terms.append(0.25 * contract('opqr,qrop', G4, G4))
        half = -contract('ijcm,im->jc', V, L)
        terms.append(-contract('jc,al,jl,ac', half, half, R, R))
        if Sinv is None:
            terms.append(contract('ju,uy,jy', half, half, R))
        else:
            terms.append(contract('ju,uv,vy,jy', half, Sinv, half, R))
        intermediate = contract('ijuq,io->juqo', V, L)
        Cx = contract('juqo,jp->uqop', intermediate, L)
        Dx = contract('ukop,kq->uopq', Y, L)
        if Sinv is None:
            terms.append(-0.5 * contract('uqop,uopq', Cx, Dx))
        else:
            terms.append(-0.5 * contract('uqop,uv,vopq', Cx, Sinv, Dx))
        terms.append(0.25 * contract('ijop,io,jp', Z, L, L))
        if H1e is not None:
            ordered = [-0.5 * contract('jl,jl,ac,ac', half, R, H1e, R), contract('jc,jl,al,ac', half, R, H1e, R)]
            terms.append(2 * sum(ordered, start=zero))
            cross = contract('mxju,xm->ju', U, L)
            terms.append(-contract('ju,uy,yj', cross, H1e if SiH is None else SiH, R))
            terms.append(-contract('uj,yu,jy', cross, H1e if HSi is None else HSi, R))
    squared = sum(terms, start=zero)
    one = constant_like(jnp.ones((), dtype=dtype), L)
    overlap = det if null is None else det * coeff_at(one, null)
    return overlap, det * coeff_at(squared, null), det * coeff_at(energy, null)
