import jax
import jax.numpy as jnp
from functools import partial


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


@partial(jax.jit, static_argnames="nfreeze", donate_argnums=(1,2))
def freeze_core(enuc, h1e_mo, h2e_mo, nfreeze):
    assert nfreeze % 2 == 0
    # times 2 because up and down; restricted core orbitals only
    ecore = 2 * jnp.einsum("ii", h1e_mo[: nfreeze // 2, : nfreeze // 2])
    ecore += 2 * jnp.einsum("iijj", h2e_mo[: nfreeze // 2, : nfreeze // 2, : nfreeze // 2, : nfreeze // 2])
    ecore -= jnp.einsum("ijij", h2e_mo[: nfreeze // 2, : nfreeze // 2, : nfreeze // 2, : nfreeze // 2])
    h1ecore = 2 * jnp.einsum("kkij->ij", h2e_mo[: nfreeze // 2, : nfreeze // 2, nfreeze // 2 :, nfreeze // 2 :])
    h1ecore -= jnp.einsum("ijki->jk", h2e_mo[: nfreeze // 2, nfreeze // 2 :, nfreeze // 2 :, : nfreeze // 2])
    enuc_ = ecore + enuc
    h1e_ = h1e_mo[nfreeze // 2 :, nfreeze // 2 :] + h1ecore
    h2e_ = h2e_mo[nfreeze // 2 :, nfreeze // 2 :, nfreeze // 2 :, nfreeze // 2 :]
    return enuc_, h1e_, h2e_


@jax.jit
def ao2mo(h, mo_coeff):
    # transform to the orthonormal MO basis
    if h.ndim == 0:
        return h
    if h.ndim == 2:
        return jnp.einsum("ij,ia,jb->ab", h, mo_coeff.conj(), mo_coeff)
    elif h.ndim == 4:
        if jnp.iscomplexobj(mo_coeff):
            raise NotImplementedError("complex mo_coeff for two-body integrals")
        return jnp.einsum("ijkl,ia,jb,kc,ld->abcd", h, mo_coeff, mo_coeff, mo_coeff, mo_coeff)
    else:
        raise NotImplementedError
