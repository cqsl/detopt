from functools import partial
import jax
import jax.numpy as jnp
from ..batching import batch_map
from ..common import _check_degree, _unpack, barrier, metric, metric_bra
from .derivatives import E1e_deriv_pair, E1e_prepare_A, E1e_prepare_B, E2e_deriv_A_first_pair, E2e_deriv_B_first_pair, _hess_pair, ol_deriv_pair
from ..keepspin import two_body, two_body_dtype


def E2e_deriv_A_first(Aconjs, Bs, H2e, rcond=0, max_degree=3, cached_batch_size=16, pair_batch_size=128, factorization='svd', S=None):
    max_degree = _check_degree(max_degree, 3)
    layouts = two_body(H2e, 'bra', two_body_dtype(H2e, Aconjs, Bs))

    def bra(Aconj):
        prepared = layouts.prepare(Aconj)
        Aconj, prepared = barrier(Aconj, prepared)
        AS = metric_bra(Aconj, S)
        return batch_map(lambda B: E2e_deriv_A_first_pair(Aconj, B, layouts, prepared, rcond, max_degree, factorization, S, AS), Bs, pair_batch_size)
    return batch_map(bra, Aconjs, cached_batch_size)


def E2e_deriv_B_first(Aconjs, Bs, H2e, rcond=0, max_degree=3, cached_batch_size=16, pair_batch_size=128, factorization='svd', S=None):
    max_degree = _check_degree(max_degree, 3)
    layouts = two_body(H2e, 'ket', two_body_dtype(H2e, Aconjs, Bs))
    ASs = None if S is None else jax.vmap(metric_bra, in_axes=(0, None))(Aconjs, S)

    def ket(B):
        prepared = layouts.prepare(B)
        B, prepared = barrier(B, prepared)

        def pair(data):
            Aconj, AS = data if ASs is not None else (data, None)
            return E2e_deriv_B_first_pair(Aconj, B, layouts, prepared, rcond, max_degree, factorization, S, AS)
        inputs = Aconjs if ASs is None else (Aconjs, ASs)
        return batch_map(pair, inputs, pair_batch_size)
    return jnp.swapaxes(batch_map(ket, Bs, cached_batch_size), 0, 1)


def E2e_deriv(Aconjs, Bs, H2e, rcond=0, max_degree=3, cached_batch_size=16, pair_batch_size=128, factorization='svd', S=None):
    max_degree = _check_degree(max_degree, 3)
    if H2e is None:
        return 0
    if Aconjs.shape[0] <= Bs.shape[0]:
        return E2e_deriv_A_first(Aconjs, Bs, H2e, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, S)
    return E2e_deriv_B_first(Aconjs, Bs, H2e, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, S)


def ol_deriv(Aconjs, Bs, rcond=0, max_degree=1, cached_batch_size=None, pair_batch_size=128, factorization='svd', S=None):
    max_degree = _check_degree(max_degree, 1)

    def bra(Aconj):
        AS = metric_bra(Aconj, S)
        return batch_map(lambda B: ol_deriv_pair(Aconj, B, rcond, max_degree, factorization, S, AS), Bs, pair_batch_size)
    return batch_map(bra, Aconjs, cached_batch_size)


def E1e_deriv(Aconjs, Bs, H1e, rcond=0, max_degree=2, cached_batch_size=None, pair_batch_size=128, factorization='svd', S=None):
    max_degree = _check_degree(max_degree, 2)
    if H1e is None:
        return 0
    if H1e.ndim == 1:
        raise NotImplementedError('E1e_deriv requires the full one-body matrix')
    HBts = jax.vmap(E1e_prepare_B, in_axes=(0, None))(Bs, H1e)

    def bra(Aconj):
        AH = E1e_prepare_A(Aconj, H1e)
        AS = metric_bra(Aconj, S)
        return batch_map(lambda data: E1e_deriv_pair(Aconj, data[0], AH, data[1], H1e, rcond, max_degree, factorization, S, AS), (Bs, HBts), pair_batch_size)
    return batch_map(bra, Aconjs, cached_batch_size)


def hess_A_first(Aconjs, Bs, H2e, H1e, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, S=None):
    layouts = two_body(H2e, 'bra', two_body_dtype(H2e, Aconjs, Bs))
    HBts = None if H1e is None else jax.vmap(E1e_prepare_B, in_axes=(0, None))(Bs, H1e)

    def bra(Aconj):
        prepared = None if layouts is None else layouts.prepare(Aconj)
        Aconj, prepared = barrier(Aconj, prepared)
        AH = None if H1e is None else E1e_prepare_A(Aconj, H1e)
        AS = metric_bra(Aconj, S)
        K_from = lambda P, L: layouts.contract(L, prepared)

        def pair(data):
            B, HBt = data if HBts is not None else (data, None)
            return _hess_pair(Aconj, B, prepared is not None, AH, HBt, H1e, Enuc, K_from, rcond, max_degree, factorization, S, AS)
        inputs = Bs if HBts is None else (Bs, HBts)
        return batch_map(pair, inputs, pair_batch_size)
    return batch_map(bra, Aconjs, cached_batch_size)


def hess_B_first(Aconjs, Bs, H2e, H1e, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, S=None):
    layouts = two_body(H2e, 'ket', two_body_dtype(H2e, Aconjs, Bs))
    AHs = None if H1e is None else jax.vmap(E1e_prepare_A, in_axes=(0, None))(Aconjs, H1e)
    ASs = None if S is None else jax.vmap(metric_bra, in_axes=(0, None))(Aconjs, S)
    cached = tuple(x for x in (Aconjs, AHs, ASs) if x is not None)

    def ket(B):
        prepared = None if layouts is None else layouts.prepare(B)
        B, prepared = barrier(B, prepared)
        HBt = None if H1e is None else E1e_prepare_B(B, H1e)

        def pair(data):
            data = data if isinstance(data, tuple) else (data,)
            Aconj = data[0]
            AH = data[1] if AHs is not None else None
            AS = data[-1] if ASs is not None else None
            K_from = lambda P, L: layouts.contract(P @ Aconj, prepared)
            return _hess_pair(Aconj, B, prepared is not None, AH, HBt, H1e, Enuc, K_from, rcond, max_degree, factorization, S, AS)
        inputs = cached[0] if len(cached) == 1 else cached
        return batch_map(pair, inputs, pair_batch_size)
    res = batch_map(ket, Bs, cached_batch_size)
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), res)


def hessian_pairs(Aconjs, Bs, H1e=None, H2e=None, Enuc=0, rcond=0, max_degree=3, cached_batch_size=16, pair_batch_size=128, factorization='svd', S=None):
    max_degree = _check_degree(max_degree, 3)
    args = (Aconjs, Bs, H2e, H1e, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, S)
    if Aconjs.shape[0] <= Bs.shape[0]:
        return hess_A_first(*args)
    return hess_B_first(*args)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def hessian(X, arrays=None, rcond=0, max_degree=3, factorization='svd', cached_batch_size=16, pair_batch_size=128):
    H1e, H2e, w, Enuc = _unpack(arrays, 'GHF hessian')
    if w is not None:
        raise NotImplementedError('w is a UHF opposite-spin operator; GHF folds it into H2e')
    Xc = X.conj()
    return hessian_pairs(Xc.X, X.X, H1e, H2e, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, metric(arrays))
