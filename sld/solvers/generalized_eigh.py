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
    cutoff = rcond * jnp.abs(s).max()
    mask = s > cutoff

    s_safe = jnp.where(mask, s, 1.0)
    s12_inv = jnp.where(mask, jax.lax.rsqrt(s_safe), 0.0)
    L = u @ jnp.diag(s12_inv)
    C = L.T.conj() @ (A @ L)
    scale = jnp.maximum(1.0, jnp.abs(C).max())
    C = C + jnp.diag(jnp.where(mask, 0.0, 1e6 * scale))
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C, UPLO="U", symmetrize_input=symmetrize_input)
    n = A.shape[0]
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


@partial(jax.jit, static_argnames=("n_zero_eigvals", "debug"))
def generalized_eigh_lowrank_decomp(U, B, n_zero_eigvals=0, rcond=1e-12, debug=False):
    # A is given by U @ U^H
    eigvals_B, u = jnp.linalg.eigh(B)
    s = eigvals_B[n_zero_eigvals:]
    u = u[:, n_zero_eigvals:]
    mask = jnp.abs(s) > rcond * jnp.abs(s).max()
    s12_inv = jax.lax.select(mask, jax.lax.rsqrt(s), jnp.zeros_like(s))
    L = u @ jnp.diag(s12_inv)

    V = L.T.conj() @ U

    rank_A = U.shape[-1]
    if rank_A == 1:
        eigenvalues = (V.T.conj() @ V).real.ravel()
        eigenvectors_transformed = V / jnp.linalg.norm(V)
    else:
        raise NotImplementedError
        # C = V@V.T.conj()
        # eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    eigenvectors_original = L @ eigenvectors_transformed
    if debug:
        return eigenvalues, eigenvectors_original, eigvals_B
    else:
        return eigenvalues, eigenvectors_original


@partial(jax.jit, static_argnames=("n_zero_eigvals", "debug"), donate_argnums=(0, 1))
def generalized_eigh_lowrank_tensor_decomp(U, B, n_zero_eigvals=0, rcond=1e-12, debug=False):
    # A is given by U * U^H

    s = U.shape[:2]
    n = np.prod(s)

    U = U.reshape(n, -1)
    B = B.reshape(n, n)

    e, v, *rest = generalized_eigh_lowrank_decomp(U, B, n_zero_eigvals=n_zero_eigvals, rcond=rcond, debug=debug)

    return e, v.T.reshape((-1,) + s), *rest


@partial(jax.jit, static_argnames="n_zero_eigvals")
def generalized_eigh_partial(Sff, Hff, Sfl, Hfl, Sll, Hll, n_zero_eigvals=0):
    Slf = flatten(Sfl).T.conj()
    Hlf = flatten(Hfl).T.conj()

    S_all_flat = jnp.block([[flatten(Sff), flatten(Sfl)], [Slf, flatten(Sll)]])
    H_all_flat = jnp.block([[flatten(Hff), flatten(Hfl)], [Hlf, flatten(Hll)]])

    e, v = generalized_eigh_lowrank(H_all_flat, S_all_flat, n_zero_eigvals=n_zero_eigvals)
    e0 = e[0]
    v0_flat = v[:, 0]

    v0_full = v0_flat[: Sff.shape[0] * Sff.shape[1]].reshape(Sff.shape[:2])
    v0_lvp = v0_flat[Sff.shape[0] * Sff.shape[1] :].reshape(Sll.shape[:2])

    return e0, v0_full, v0_lvp
