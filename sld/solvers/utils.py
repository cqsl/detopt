import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import reshard

from .nullspace_proj import proj_out_kernel


@partial(jax.jit, static_argnames=("ne", "trafo"))
def preprocess_cholesky(H, S, ne, trafo=True):
    A, B, P, _ = proj_out_kernel(H, S, ne)

    A = jax.lax.collapse(jax.lax.collapse(A, 0, 2), 1, 3)
    B = jax.lax.collapse(jax.lax.collapse(B, 0, 2), 1, 3)

    # gather all of it and run cholesky serially
    B = reshard(B, jax.P())
    L = jax.lax.linalg.cholesky(B, symmetrize_input=False)
    if not trafo:
        return A, P, L

    # TODO update the rest for sharding
    I = jnp.eye(L.shape[0], dtype=L.dtype)
    Linv = jax.lax.linalg.triangular_solve(L, I, lower=True)  # TODO conj
    C = jnp.einsum("ij,jk,lk->il", Linv, A, Linv.conj())  # Linv @ A @ Linv.T.conj()
    return C, P, Linv.T.conj()


@partial(jax.jit, static_argnames="n")
def postprocess(v, P, *, L=None, Linv=None, n=None):
    if Linv is not None:
        v = jnp.einsum("ij,nj->ni", Linv, v)
    else:
        v = jax.lax.linalg.triangular_solve(L, v, lower=True, conjugate_a=True)
    if P is None:
        return v.reshape(v.shape[0], -1, n)
    return jnp.einsum("naj,aij->nai", v.reshape(-1, P.shape[0], P.shape[2]), P)


def preprocess_guess(v, P, L):
    v = jnp.einsum("naj,aji->nai", v, P.conj())
    v = v.reshape(v.shape[0], -1)
    return jnp.einsum("ji,aj->ai", L.conj(), v)


@jax.jit
def matvec(A, L, x):
    tmp1 = jax.lax.linalg.triangular_solve(L, x, lower=True, conjugate_a=True)
    tmp2 = A @ tmp1
    return jax.lax.linalg.triangular_solve(L, tmp2, lower=True, transpose_a=True)


@jax.jit
@partial(jax.smap, out_axes=1, axis_name="i")
def _sol1(L, A):
    return jax.lax.linalg.triangular_solve(L, A, lower=True, left_side=True)


@jax.jit
@partial(jax.smap, out_axes=0, axis_name="i")
def _sol2(L, tmp):
    return jax.lax.linalg.triangular_solve(L, tmp, lower=True, transpose_a=True, conjugate_a=True)


@jax.jit
def compute_C_sh(L, A):
    tmp = _sol2(L, A)
    tmp = reshard(tmp, jax.P(None, "i"))
    return _sol1(L, tmp)


def flatten(x):
    return jax.lax.collapse(jax.lax.collapse(x, 0, 2), 1, 3)
