import jax
import jax.numpy as jnp


@jax.jit
def particle_hole_trafo(const, hij, hijkl):
    if hijkl is None:
        const_ph = const + 2 * jnp.diagonal(hij).sum()
        hij_ph = -hij.T
        return const_ph, hij_ph, None
    else:
        const_ph = const + 2 * jnp.diagonal(hij).sum() + 4 * jnp.einsum("iijj", hijkl) - 2 * jnp.einsum("ijji", hijkl)
        hij_ph = -hij.T
        hij_ph = hij_ph - 2 * (jnp.einsum("iljj->li", hijkl) + jnp.einsum("iijk->kj", hijkl))
        hij_ph = hij_ph + (jnp.einsum("ilji->lj", hijkl) + jnp.einsum("ijjk->ki", hijkl))
        hijkl_ph = hijkl
        return const_ph, hij_ph, hijkl_ph


@jax.jit
def ao2mo(h, mo_coeff):
    # use mo basis which is orthogonal
    # alternatively we could use the sqrtm of the overlap matrix
    # or, to avoid the O(n_orb^5) trafo of h2e, include it in all the contractions
    if h.ndim == 0:
        return h
    if h.ndim == 2:
        return jnp.einsum("ij,ia,jb->ab", h, mo_coeff, mo_coeff)
    elif h.ndim == 4:
        return jnp.einsum("ijkl,ia,jb,kc,ld->abcd", h, mo_coeff, mo_coeff, mo_coeff, mo_coeff)
    else:
        raise NotImplementedError
