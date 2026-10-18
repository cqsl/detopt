from functools import partial

import jax
import jax.numpy as jnp
from jax import P
from jax.sharding import reshard

from sld.mels import compute_mels
from sld.mels._src.hybrid import uhf as hybrid_uhf


@partial(jax.jit, static_argnames="sum_f")
def integrate_and_assemble_all(v0_full_normalized, Sff, Hff, Sfl, Hfl, Sll, Hll, sum_f=False):
    if sum_f:
        # sum over all dets representing a single state
        ab = "b"
        ab2 = ""
        idx = None
        idx2 = None, None
    else:
        ab = ab2 = "ab"
        idx = idx2 = Ellipsis
    # match the sharding of the determinant axes of each operator, the results are replicated anyway
    v = lambda M, axis: reshard(v0_full_normalized, P(jax.typeof(M).sharding.spec[axis]))
    # x is a dummy dimension
    Sfl = jnp.einsum(f"aibx,ai->{ab}", Sfl, v(Sfl, 0).conj(), out_sharding=P())[idx]
    Hfl = jnp.einsum(f"aibx,ai->{ab}", Hfl, v(Hfl, 0).conj(), out_sharding=P())[idx]
    Sff = jnp.einsum(f"aibj,ai,bj->{ab2}", Sff, v(Sff, 0).conj(), v(Sff, 2), out_sharding=P())[idx2]
    Hff = jnp.einsum(f"aibj,ai,bj->{ab2}", Hff, v(Hff, 0).conj(), v(Hff, 2), out_sharding=P())[idx2]
    Sll = jax.lax.collapse(jax.lax.collapse(Sll, 0, 2), 1, 3)
    Hll = jax.lax.collapse(jax.lax.collapse(Hll, 0, 2), 1, 3)

    Sfl, Hfl, Sff, Hff, Sll, Hll = map(partial(reshard, out_shardings=jax.P()), (Sfl, Hfl, Sff, Hff, Sll, Hll))

    S_all = jnp.block([[Sff, Sfl], [Sfl.T.conj(), Sll]])
    H_all = jnp.block([[Hff, Hfl], [Hfl.T.conj(), Hll]])
    return S_all, H_all


def expval_blocks(impls, X, hamiltonian, batch_size, backend, rcond, max_degree, factorization, cached_batch_size, pair_batch_size):
    if backend in ("hybrid", "hybrid_svd"):
        if jax.device_count() == 1:
            S, H = hybrid_uhf.expval(X, X, arrays=hamiltonian, rcond=rcond, max_degree=max_degree, factorization=factorization, cached_batch_size=cached_batch_size, pair_batch_size=pair_batch_size)
        else:
            S, H = hybrid_uhf.expval_sharded(reshard(X, P("i")), X, arrays=hamiltonian, rcond=rcond, max_degree=max_degree, factorization=factorization, cached_batch_size=cached_batch_size, pair_batch_size=pair_batch_size)
        return S[:, None, :, None], H[:, None, :, None]
    S, H, *_ = compute_mels(impls.expval, X, reshard(X, P("i")), arrays=hamiltonian, batch_size=batch_size)
    return S, H
