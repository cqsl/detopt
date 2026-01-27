import jax
import jax.numpy as jnp
from functools import partial

import numpy as np

from .utils import flatten


@partial(jax.jit, static_argnames=("n_zero_eigvals", "debug", "symmetrize_input"))
def generalized_eigh_lowrank(A, B, n_zero_eigvals=0, rcond=1e-12, debug=False, symmetrize_input=True):
    eigvals_B, u = jnp.linalg.eigh(B, UPLO="U", symmetrize_input=symmetrize_input)
    s = eigvals_B[n_zero_eigvals:]
    u = u[:, n_zero_eigvals:]
    mask = jnp.abs(s) > rcond * jnp.abs(s).max()
    s12_inv = jax.lax.select(mask, jax.lax.rsqrt(s), jnp.zeros_like(s))
    L = u @ jnp.diag(s12_inv)
    C = L.T.conj() @ (A @ L)
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C, UPLO="U", symmetrize_input=symmetrize_input)
    n = A.shape[0]
    # TODO why is the sign here needed?
    sign = 2 * (jnp.arange(n - n_zero_eigvals) % 2) - 1
    eigenvectors_original = L @ eigenvectors_transformed @ jnp.diag(sign)
    if debug:
        return eigenvalues, eigenvectors_original, eigvals_B
    else:
        return eigenvalues, eigenvectors_original


@partial(jax.jit, static_argnames=("n_zero_eigvals", "debug", "symmetrize_input"), donate_argnums=(0, 1))
def generalized_eigh_lowrank_tensor(A, B, n_zero_eigvals=0, rcond=1e-12, debug=False, symmetrize_input=True):
    s = A.shape[:2]

    A = jax.lax.collapse(jax.lax.collapse(A, 0, 2), 1, 3)
    B = jax.lax.collapse(jax.lax.collapse(B, 0, 2), 1, 3)

    e, v, *rest = generalized_eigh_lowrank(A, B, n_zero_eigvals=n_zero_eigvals, rcond=rcond, debug=debug, symmetrize_input=symmetrize_input)

    return e, v.T.reshape((-1,) + s), *rest


@partial(jax.jit, static_argnames="n_zero_eigvals")
def generalized_eigh_partial(Sff, Hff, Sfl, Hfl, Sll, Hll, n_zero_eigvals=0):
    # TODO if we adapt the solver to only use the lower triag part we can replace this with zeros
    Slf = flatten(Sfl).T.conj()
    Hlf = flatten(Sfl).T.conj()

    S_all_flat = jnp.block([[flatten(Sff), flatten(Sfl)], [Slf, flatten(Sll)]])
    H_all_flat = jnp.block([[flatten(Hff), flatten(Hfl)], [Hlf, flatten(Hll)]])

    e, v = generalized_eigh_lowrank(H_all_flat, S_all_flat, n_zero_eigvals=n_zero_eigvals)
    e0 = e[0]
    v0_flat = v[:, 0]

    v0_full = v0_flat[: Sff.shape[0] * Sff.shape[1]].reshape(Sff.shape[:2])
    v0_lvp = v0_flat[Sff.shape[0] * Sff.shape[1] :].reshape(Sll.shape[:2])

    return e0, v0_full, v0_lvp
