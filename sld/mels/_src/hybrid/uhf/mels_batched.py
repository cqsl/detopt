from functools import partial
import jax
import jax.numpy as jnp
from jax import P
from ...pairwise.common import antisymmetrized
from ..algebra import result_type
from ..batching import _check_replicated, _check_sharding, _mesh, batch_map, tick
from ..common import _check_degree, _unpack, barrier, metric, metric_bra, prepare_Htt, w_A_other, split_spin, squared_metric
from .mels import _expval_pair, _expval_sq_pair, _expval_sq_pair_prepared, _expval_sq_prepare_A, H_A


def expval_pairs(Aconju, Aconjd, Bu, Bd, H1e=None, H2e=None, w=None, Enuc=0, rcond=1e-10, max_degree=2, cached_batch_size=16, pair_batch_size=128, factorization='svd', U=None, S=None, progress=None):
    max_degree = _check_degree(max_degree, 2)
    if U is not None and S is not None:
        raise NotImplementedError('the hybrid rotation overlap has no nonorthogonal form')
    H1u, H1d = split_spin(H1e)
    Htt = prepare_Htt(H2e, Aconju, Aconjd, Bu, Bd)

    def bra(data):
        Au, Ad = data
        HAu = H_A(Au, Htt)
        HAd = H_A(Ad, Htt)
        Au, HAu = barrier(Au, HAu)
        Ad, HAd = barrier(Ad, HAd)
        wAo = w_A_other(Ad, w)
        ASu = metric_bra(Au, S)
        ASd = metric_bra(Ad, S)
        result = batch_map(lambda kets: _expval_pair(Au, Ad, kets[0], kets[1], HAu, HAd, wAo, H1u, H1d, Enuc, rcond, max_degree, factorization, U, ASu, ASd), (Bu, Bd), pair_batch_size)
        tick(progress)
        return result
    return batch_map(bra, (Aconju, Aconjd), cached_batch_size)


def expval_sq_pairs(Aconju, Aconjd, Bu, Bd, H1e=None, H2e=None, w=None, Enuc=0, rcond=1e-10, max_degree=4, cached_batch_size=16, pair_batch_size=128, factorization='svd', S=None, progress=None):
    max_degree = _check_degree(max_degree, 4)
    H1u, H1d = split_spin(H1e)
    Htt = None if H2e is None else antisymmetrized(H2e)[2]
    dtype = result_type(Aconju, Aconjd, Bu, Bd, *(part for part in (H1u, H1d, Htt, w) if part is not None))
    cast = lambda part: None if part is None else jnp.asarray(part, dtype)
    H1u, H1d = cast(H1u), cast(H1d)
    S = None if S is None else jnp.asarray(S, dtype)
    sq_up = squared_metric(S, H1u)
    sq_down = sq_up if H1d is H1u else squared_metric(S, H1d)

    if S is None:
        def bra(data):
            Au, Ad = data
            prepared = _expval_sq_prepare_A(Au, Ad, Htt, w, H1u is not None or H1d is not None)
            (Au, Ad), prepared = barrier((Au, Ad), prepared)
            result = batch_map(lambda kets: _expval_sq_pair_prepared(Au, Ad, kets[0], kets[1], H1u, H1d, prepared, Enuc, dtype, rcond, max_degree, factorization), (Bu, Bd), pair_batch_size)
            tick(progress)
            return result
        return batch_map(bra, (Aconju, Aconjd), cached_batch_size)

    Htt, w = cast(Htt), cast(w)

    def bra(data):
        Au, Ad = data
        ASu = metric_bra(Au, S)
        ASd = metric_bra(Ad, S)
        result = batch_map(lambda kets: _expval_sq_pair(Au, Ad, kets[0], kets[1], H1u, H1d, Htt, w, Enuc, rcond, max_degree, factorization, sq_up, sq_down, ASu, ASd), (Bu, Bd), pair_batch_size)
        tick(progress)
        return result
    return batch_map(bra, (Aconju, Aconjd), cached_batch_size)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size', 'progress'))
def expval_sq(X, Y, arrays=None, rcond=1e-10, max_degree=4, factorization='svd', cached_batch_size=16, pair_batch_size=128, progress=None):
    H1e, H2e, w, Enuc = _unpack(arrays, 'UHF squared expectation value')
    Xc = X.conj()
    return expval_sq_pairs(Xc.Xu, Xc.Xd, Y.Xu, Y.Xd, H1e, H2e, w, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, metric(arrays), progress)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size', 'progress'))
def expval(X, Y, arrays=None, rcond=1e-10, max_degree=2, factorization='svd', cached_batch_size=16, pair_batch_size=128, progress=None):
    H1e, H2e, w, Enuc, U = _unpack(arrays, 'UHF expectation value', rotation=True)
    Xc = X.conj()
    return expval_pairs(Xc.Xu, Xc.Xd, Y.Xu, Y.Xd, H1e, H2e, w, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, U, metric(arrays), progress)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def expval_sharded(X, Y, arrays=None, rcond=1e-10, max_degree=2, factorization='svd', cached_batch_size=16, pair_batch_size=128):
    mesh, nshards = _mesh()
    if nshards == 1:
        return expval(X, Y, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size)
    hamiltonian = _check_replicated(*_unpack(arrays, 'UHF expectation value', rotation=True), metric(arrays))
    _check_sharding(X, P('i'), 'X')
    _check_sharding(Y, P(), 'Y')
    Xc = X.conj()

    def rows(Xu, Xd, Yu, Yd, hamiltonian):
        H1e, H2e, w, Enuc, U, S = hamiltonian
        return expval_pairs(Xu, Xd, Yu, Yd, H1e, H2e, w, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, U, S)
    evaluate = jax.shard_map(rows, mesh=mesh, in_specs=(P('i'), P('i'), P(), P(), P()), out_specs=(P('i'), P('i')))
    return evaluate(Xc.Xu, Xc.Xd, Y.Xu, Y.Xd, hamiltonian)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def expval_sq_sharded(X, Y, arrays=None, rcond=1e-10, max_degree=4, factorization='svd', cached_batch_size=16, pair_batch_size=128):
    mesh, nshards = _mesh()
    if nshards == 1:
        return expval_sq(X, Y, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size)
    hamiltonian = _check_replicated(*_unpack(arrays, 'UHF squared expectation value'), metric(arrays))
    _check_sharding(X, P('i'), 'X')
    _check_sharding(Y, P(), 'Y')
    Xc = X.conj()

    def rows(Xu, Xd, Yu, Yd, hamiltonian):
        H1e, H2e, w, Enuc, S = hamiltonian
        return expval_sq_pairs(Xu, Xd, Yu, Yd, H1e, H2e, w, Enuc, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, S)
    evaluate = jax.shard_map(rows, mesh=mesh, in_specs=(P('i'), P('i'), P(), P(), P()), out_specs=(P('i'), P('i'), P('i')))
    return evaluate(Xc.Xu, Xc.Xd, Y.Xu, Y.Xd, hamiltonian)
