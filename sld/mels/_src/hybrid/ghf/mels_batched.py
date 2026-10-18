from functools import partial
import jax
import jax.numpy as jnp
from ...pairwise.common import antisymmetrized
from ..algebra import result_type
from ..batching import batch_map, tick
from ..common import _check_degree, _unpack, barrier, metric, metric_bra, squared_metric
from .derivatives import E1e_prepare_A
from ..keepspin import KeepSpinTwoBody, is_keepspin, keepspin_coefficients, two_body, two_body_dtype
from .mels import _expval_pair, _expval_sq_pair_prepared, _expval_sq_prepare_A


def expval_pairs(Aconjs, Bs, H1e=None, H2e=None, Enuc=0, rcond=0, max_degree=2, cached_batch_size=16, pair_batch_size=128, factorization='svd', S=None, progress=None):
    max_degree = _check_degree(max_degree, 2)
    layouts = two_body(H2e, 'bra', two_body_dtype(H2e, Aconjs, Bs))

    def bra(Aconj):
        prepared = None if layouts is None else layouts.prepare(Aconj)
        Aconj, prepared = barrier(Aconj, prepared)
        AH = None if H1e is None else E1e_prepare_A(Aconj, H1e)
        AS = metric_bra(Aconj, S)
        K_from = lambda P, L: layouts.contract(L, prepared)
        result = batch_map(lambda B: _expval_pair(Aconj, B, prepared is not None, AH, Enuc, K_from, rcond, max_degree, factorization, AS), Bs, pair_batch_size)
        tick(progress)
        return result
    return batch_map(bra, Aconjs, cached_batch_size)


def expval_sq_pairs(Aconjs, Bs, H1e=None, H2e=None, Enuc=0, rcond=1e-10, max_degree=4, cached_batch_size=16, pair_batch_size=128, factorization='svd', S=None, progress=None):
    if max_degree < 0 or max_degree > 4:
        raise ValueError('max_degree must be between 0 and 4 for the GHF squared expectation value')
    if H1e is None and H2e is None:
        overlap, _ = expval_pairs(Aconjs, Bs, Enuc=0, rcond=rcond, max_degree=min(max_degree, 1), cached_batch_size=cached_batch_size, pair_batch_size=pair_batch_size, factorization=factorization, S=S)
        return overlap, Enuc * Enuc * overlap, Enuc * overlap
    if H2e is None:
        max_degree = min(max_degree, 2)
    keepspin = is_keepspin(H2e)
    coefficients = () if H2e is None else keepspin_coefficients(H2e) if keepspin else (antisymmetrized(H2e)[2],)
    dtype = result_type(Aconjs, Bs, *coefficients, *(() if H1e is None else (H1e,)))
    H1e = None if H1e is None else jnp.asarray(H1e, dtype)
    Htt = None if H2e is None else KeepSpinTwoBody(coefficients) if keepspin else coefficients[0]
    S = None if S is None else jnp.asarray(S, dtype)
    sq_metric = squared_metric(S, H1e)

    def bra(Aconj):
        prepared = None if Htt is None else _expval_sq_prepare_A(Aconj, Htt, H1e is not None, sq_metric[0])
        Aconj, prepared = barrier(Aconj, prepared)
        AS = metric_bra(Aconj, S)
        result = batch_map(lambda B: _expval_sq_pair_prepared(Aconj, B, H1e, prepared, Enuc, dtype, rcond, max_degree, factorization, sq_metric, AS), Bs, pair_batch_size)
        tick(progress)
        return result
    return batch_map(bra, Aconjs, cached_batch_size)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size', 'progress'))
def expval_sq(X, Y, arrays=None, rcond=1e-10, max_degree=4, factorization='svd', cached_batch_size=16, pair_batch_size=128, progress=None):
    H1e, H2e, w, Enuc = _unpack(arrays, 'GHF squared expectation value')
    if w is not None:
        raise NotImplementedError('w is a UHF opposite-spin operator; GHF folds it into H2e')
    return expval_sq_pairs(X.conj().X, Y.X, H1e, H2e, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, metric(arrays), progress)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size', 'progress'))
def expval(X, Y, arrays=None, rcond=0, max_degree=2, factorization='svd', cached_batch_size=16, pair_batch_size=128, progress=None):
    H1e, H2e, w, Enuc = _unpack(arrays, 'GHF expectation value')
    if w is not None:
        raise NotImplementedError('w is a UHF opposite-spin operator; GHF folds it into H2e')
    return expval_pairs(X.conj().X, Y.X, H1e, H2e, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, metric(arrays), progress)
