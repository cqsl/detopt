import jax
import jax.numpy as jnp
from ..pairwise.common import antisymmetrized
from ..pairwise.unstable.mels import inv_and_det
from .algebra import Polynomial, contract, result_type, trace


def _unpack(arrays, what, rotation=False):
    arrays = {} if arrays is None else arrays
    if arrays.get('U_onsite', None) is not None:
        raise NotImplementedError(f'the hybrid {what} does not support onsite operators')
    U = arrays.get('U', None)
    if U is not None and (not rotation):
        raise NotImplementedError(f'the hybrid {what} does not support projected operators')
    parts = (arrays.get('H1e', None), arrays.get('H2e', None), arrays.get('w', None), arrays.get('Enuc', 0))
    return parts + (U,) if rotation else parts


def metric(arrays):
    S = None if arrays is None else arrays.get('S', None)
    return None if S is None else jnp.asarray(S)


def metric_bra(Aconj, S):
    return Aconj if S is None else Aconj @ S


def metric_factor(L, S):
    return L if S is None else S @ L


def squared_metric(S, H1e):
    # (Sinv, Sinv H1e, H1e Sinv^T)
    if S is None:
        return None, None, None
    Sinv = jnp.linalg.inv(S)
    if H1e is None:
        return Sinv, None, None
    return Sinv, Sinv @ H1e, H1e @ Sinv.T


def barrier(*xs):
    leaves, treedef = jax.tree.flatten(xs)
    if not leaves:
        return xs
    return jax.tree.unflatten(treedef, jax.lax.optimization_barrier(tuple(leaves)))


def normalize_index(index, size):
    return jnp.where(index < 0, index + size, index)


def remove_row(X, index):
    return jnp.delete(X, normalize_index(index, X.shape[0]), axis=0, assume_unique_indices=True)


def projector(L, Aconj, dtype, S=None):
    identity = jnp.eye(Aconj.shape[1], dtype=dtype) if S is None else S
    return identity - L @ Aconj


def inverse_polynomial(Aconj, B, rcond, max_degree, factorization, AS=None):
    overlap = (Aconj if AS is None else AS) @ B.T
    if factorization == 'lu':
        if max_degree != 0:
            raise ValueError('LU factorization requires max_degree=0')
        inverse, determinant = inv_and_det(overlap)
        return inverse, determinant, None
    if factorization != 'svd':
        raise ValueError("factorization must be 'lu' or 'svd'")
    X, sigma, YH = jnp.linalg.svd(overlap)
    # initial=0 for an empty sector
    threshold = rcond * jnp.max(jnp.abs(sigma), initial=0)
    is_zero = jnp.abs(sigma) <= threshold
    safe_sigma = jnp.where(is_zero, jnp.ones_like(sigma), sigma)
    weights = [jnp.where(is_zero, jnp.zeros_like(sigma), 1 / safe_sigma)]
    if max_degree >= 1:
        weights.append(is_zero.astype(sigma.dtype))
    coefficients = tuple(contract('ki,i,il->kl', YH.conj().T, weight, X.conj().T) for weight in weights)
    determinant = jnp.linalg.det(X) * jnp.linalg.det(YH) * jnp.where(is_zero, jnp.ones_like(sigma), sigma).prod()
    return Polynomial(coefficients, max_degree), determinant, jnp.count_nonzero(is_zero)


def _check_degree(max_degree, limit):
    if max_degree < 0 or max_degree > limit:
        raise ValueError(f'max_degree must be between 0 and {limit}')
    return max_degree


def _two_body_energy(Aconj, L, K):
    return 0.5 * trace(Aconj @ K @ L)


def split_spin(H1e):
    return H1e if isinstance(H1e, tuple) else (H1e, H1e)


def reorder_Htt(Htt, first=1, dtype=None):
    if Htt is None:
        return None
    return jnp.moveaxis(Htt, 3, first).astype(Htt.dtype if dtype is None else dtype)


def prepare_Htt(H2e, *operands):
    if H2e is None:
        return None
    Htt = antisymmetrized(H2e)[2]
    return reorder_Htt(Htt, 1, result_type(*operands, Htt))


def w_A_other(Aconj, w):
    return None if w is None else contract('egba,qg->ebaq', w, Aconj)


def w_A(Aconj, w):
    return None if w is None else contract('egba,qa->qegb', w, Aconj)


def _factors(Aconj, B, rcond=0, max_degree=0, factorization='lu', AS=None):
    P, determinant, null = inverse_polynomial(Aconj, B, rcond, max_degree, factorization, AS)
    return P, B.T @ P, determinant, null


def _uses_enuc(Enuc):
    return not (isinstance(Enuc, (int, float)) and Enuc == 0)
