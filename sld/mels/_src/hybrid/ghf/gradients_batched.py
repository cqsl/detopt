from functools import partial
import jax
import jax.numpy as jnp
from ..batching import batch_map
from ..common import _check_degree, _unpack, barrier, metric, metric_bra, normalize_index
from .gradients import grad_rectangular_A_first_pair, grad_rectangular_B_first_pair, grad_rectangular_prepare_A, grad_rectangular_prepare_B
from ..keepspin import two_body, two_body_dtype


def grad_rectangular_A_first(Aconjs, Bs, bra_indices, ket_indices, H2e, H1e, H1e_extra, same_spin_scale, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, S=None):
    layouts = two_body(H2e, 'bra', two_body_dtype(H2e, Aconjs, Bs))

    def bra(data):
        Aconj, bra_index = data
        Ah, prepared = grad_rectangular_prepare_A(Aconj, bra_index, layouts)
        Ah, prepared = barrier(Ah, prepared)
        AS = metric_bra(Ah, S)
        return batch_map(lambda x: grad_rectangular_A_first_pair(Ah, x[0], bra_index, x[1], layouts, prepared, H1e, H1e_extra, same_spin_scale, rcond, max_degree, factorization, S, AS), (Bs, ket_indices), pair_batch_size)
    return batch_map(bra, (Aconjs, bra_indices), cached_batch_size)


def grad_rectangular_B_first(Aconjs, Bs, bra_indices, ket_indices, H2e, H1e, H1e_extra, same_spin_scale, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, S=None):
    layouts = two_body(H2e, 'ket', two_body_dtype(H2e, Aconjs, Bs))

    def ket(data):
        B, ket_index = data
        ket_index = normalize_index(ket_index, B.shape[0])
        Bh, prepared = grad_rectangular_prepare_B(B, ket_index, layouts)
        Bh, prepared = barrier(Bh, prepared)
        return batch_map(lambda x: grad_rectangular_B_first_pair(x[0], Bh, x[1], ket_index, layouts, prepared, B[ket_index], H1e, H1e_extra, same_spin_scale, rcond, max_degree, factorization, S), (Aconjs, bra_indices), pair_batch_size)
    result = batch_map(ket, (Bs, ket_indices), cached_batch_size)
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), result)


def gradient_pairs(Aconjs, Bs, bra_indices, ket_indices, H1e=None, H2e=None, H1e_extra=None, same_spin_scale=1, rcond=0, max_degree=3, cached_batch_size=16, pair_batch_size=128, factorization='svd', S=None):
    max_degree = _check_degree(max_degree, 3)
    args = (Aconjs, Bs, bra_indices, ket_indices, H2e, H1e, H1e_extra, same_spin_scale, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, S)
    if Aconjs.shape[0] <= Bs.shape[0]:
        return grad_rectangular_A_first(*args)
    return grad_rectangular_B_first(*args)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def gradient(X, Y, arrays=None, rcond=0, max_degree=3, factorization='svd', cached_batch_size=16, pair_batch_size=128):
    H1e, H2e, w, _ = _unpack(arrays, 'GHF gradient')
    if w is not None:
        raise NotImplementedError('w is a UHF opposite-spin operator; GHF folds it into H2e')
    if X.i is None or Y.i is None:
        raise ValueError('the GHF gradient needs a selected orbital row on both sides')
    Xc = X.conj()
    return gradient_pairs(Xc.X, Y.X, Xc.i, Y.i, H1e, H2e, None, 1, rcond, max_degree, cached_batch_size, pair_batch_size, factorization, metric(arrays))
