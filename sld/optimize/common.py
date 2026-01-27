import jax
import jax.numpy as jnp
from jax.sharding import reshard
from functools import partial


@jax.jit
def integrate_and_assemble_all(v0_full_normalized, Sff, Hff, Sfl, Hfl, Sll, Hll):
    # x is dummy dim, remove it
    if True:
        ab = ab2 = "ab"
        idx = idx2 = Ellipsis
    # x is a dummy dim here
    Sfl = jnp.einsum(f"aibx,ai->{ab}", Sfl, v0_full_normalized.conj())[idx]
    Hfl = jnp.einsum(f"aibx,ai->{ab}", Hfl, v0_full_normalized.conj())[idx]
    Sff = jnp.einsum(f"aibj,ai,bj->{ab2}", Sff, v0_full_normalized.conj(), v0_full_normalized)[idx2]
    Hff = jnp.einsum(f"aibj,ai,bj->{ab2}", Hff, v0_full_normalized.conj(), v0_full_normalized)[idx2]
    Sll = jax.lax.collapse(jax.lax.collapse(Sll, 0, 2), 1, 3)
    Hll = jax.lax.collapse(jax.lax.collapse(Hll, 0, 2), 1, 3)

    Sfl, Hfl, Sff, Hff, Sll, Hll = map(partial(reshard, out_shardings=jax.P()), (Sfl, Hfl, Sff, Hff, Sll, Hll))

    S_all = jnp.block([[Sff, Sfl], [Sfl.T.conj(), Sll]])
    H_all = jnp.block([[Hff, Hfl], [Hfl.T.conj(), Hll]])
    return S_all, H_all
