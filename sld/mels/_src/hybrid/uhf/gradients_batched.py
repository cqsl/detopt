from functools import partial
import jax
import jax.numpy as jnp
from jax import P
from ..algebra import contract
from ..batching import _check_replicated, _check_sharding, _mesh, batch_map
from ..common import _check_degree, _unpack, barrier, metric, metric_bra, normalize_index, prepare_Htt, w_A_other, remove_row, split_spin
from .gradients import _gradient_pair
from .mels import H_A


def _gradient_for_spin(A, Ao, B, Bo, bra_index, ket_index, Htt, wAo, H1, H1o, Enuc, rcond, max_degree, factorization, pair_batch_size, S=None):
    bra_index = normalize_index(bra_index, A.shape[0])
    ket_index = normalize_index(ket_index, B.shape[1])
    Ah = remove_row(A, bra_index)
    HA = H_A(Ah, Htt)
    HAo = H_A(Ao, Htt)
    Ah, HA = barrier(Ah, HA)
    Ao, HAo = barrier(Ao, HAo)
    sign = 1 - 2 * ((bra_index + ket_index) % 2)
    AS = metric_bra(Ah, S)
    ASo = metric_bra(Ao, S)
    return batch_map(lambda Bs: _gradient_pair(Ah, Ao, Bs[0], Bs[1], ket_index, HA, HAo, wAo, H1, H1o, Enuc, sign, rcond, max_degree, factorization, S, AS, ASo), (B, Bo), pair_batch_size)


def _compute_grad(X, Y, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, ket_weights):
    max_degree = _check_degree(max_degree, 3)
    if X.s is None or X.i is None or Y.i is not None:
        raise ValueError('the hybrid gradient requires selected UHF bras and unselected UHF kets')
    H1e, H2e, w, Enuc = _unpack(arrays, 'UHF gradient')
    S = metric(arrays)
    H1u, H1d = split_spin(H1e)
    Xc = X.conj()
    Htt = prepare_Htt(H2e, Xc.Xu, Xc.Xd, Y.Xu, Y.Xd)
    balanced = X.Xu.shape[1:] == X.Xd.shape[1:] and Y.Xu.shape[1:] == Y.Xd.shape[1:]

    def bra(data):
        Au, Ad, spin, indices = data
        if balanced:
            select = lambda a, b: jnp.where(spin == 0, a, b)
            wAo = w_A_other(select(Ad, Au), w)
            index = select(indices[0], indices[1])
            overlaps, value = _gradient_for_spin(select(Au, Ad), select(Ad, Au), select(Y.Xu, Y.Xd), select(Y.Xd, Y.Xu), index, index, Htt, wAo, H1u if H1u is H1d else select(H1u, H1d), H1d if H1u is H1d else select(H1d, H1u), Enuc, rcond, max_degree, factorization, pair_batch_size, S)
        else:
            Su, Hu = _gradient_for_spin(Au, Ad, Y.Xu, Y.Xd, indices[0], indices[0], Htt, w_A_other(Ad, w), H1u, H1d, Enuc, rcond, max_degree, factorization, pair_batch_size, S)
            Sd, Hd = _gradient_for_spin(Ad, Au, Y.Xd, Y.Xu, indices[1], indices[1], Htt, w_A_other(Au, w), H1d, H1u, Enuc, rcond, max_degree, factorization, pair_batch_size, S)
            overlaps = jnp.where(spin == 0, Su, Sd)
            value = jnp.where(spin == 0, Hu, Hd)
        if ket_weights is None:
            return overlaps, value
        return contract('qoi,qs->os', overlaps, ket_weights), contract('qoi,qs->os', value, ket_weights)
    result = batch_map(bra, (Xc.Xu, Xc.Xd, Xc.s, Xc.i), cached_batch_size)
    if ket_weights is not None:
        return result
    return jax.tree.map(lambda value: value.transpose(0, 2, 1, 3), result)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def gradient(X, Y, arrays=None, rcond=1e-10, max_degree=3, factorization='svd', cached_batch_size=1, pair_batch_size=128):
    return _compute_grad(X, Y, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, None)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def gradient_weighted(X, Y, ket_weights, arrays=None, rcond=1e-10, max_degree=3, factorization='svd', cached_batch_size=1, pair_batch_size=128):
    if ket_weights.ndim != 2 or ket_weights.shape[0] != Y.ndet:
        raise ValueError('ket_weights must have shape (len(Y), nstates)')
    return _compute_grad(X, Y, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, ket_weights)


def _gradient_sharded(X, Y, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, ket_weights, what):
    mesh, nshards = _mesh()
    if nshards == 1:
        return _compute_grad(X, Y, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, ket_weights)
    hamiltonian = _check_replicated(*_unpack(arrays, what), metric(arrays))
    _check_sharding(X, P('i'), 'X')
    _check_sharding(Y, P(), 'Y')
    if ket_weights is not None:
        _check_sharding(ket_weights, P(), 'ket_weights')

    def rows(X, Y, ket_weights, hamiltonian):
        H1e, H2e, w, Enuc, S = hamiltonian
        return _compute_grad(X, Y, {'H1e': H1e, 'H2e': H2e, 'w': w, 'Enuc': Enuc, 'S': S}, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, ket_weights)

    evaluate = jax.shard_map(
        rows,
        mesh=mesh,
        in_specs=(P('i'), P(), P(), P()),
        out_specs=(P('i'), P('i')),
    )
    return evaluate(X, Y, ket_weights, hamiltonian)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def gradient_sharded(X, Y, arrays=None, rcond=1e-10, max_degree=3, factorization='svd', cached_batch_size=1, pair_batch_size=128):
    return _gradient_sharded(X, Y, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, None, 'UHF gradient')


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def gradient_weighted_sharded(X, Y, ket_weights, arrays=None, rcond=1e-10, max_degree=3, factorization='svd', cached_batch_size=1, pair_batch_size=128):
    if ket_weights.ndim != 2 or ket_weights.shape[0] != Y.ndet:
        raise ValueError('ket_weights must have shape (len(Y), nstates)')
    return _gradient_sharded(X, Y, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, ket_weights, 'UHF gradient')
