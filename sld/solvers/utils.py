import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import reshard
from jax.tree_util import Partial

from .nullspace_proj import _uses_mesh_axis, proj_out_kernel


def projection_rank(norb, nelec, spin):
    if not isinstance(nelec, tuple):
        return nelec, None
    neu, ned = nelec
    if neu == ned:
        return neu, None
    if spin is None:
        raise ValueError("spin is required if nelec[0] != nelec[1]")
    r_min = min(neu, ned) - 1
    width = norb - r_min
    r_per_det = jnp.where(spin == 0, neu - 1, ned - 1)
    return r_min + 1, (jnp.arange(width)[None, :] >= (r_per_det - r_min)[:, None]).ravel()


def regularize(B, reg_mask):
    if reg_mask is None:
        return B, None
    scale = jnp.maximum(1.0, jnp.abs(B).max())
    reg = jnp.where(reg_mask, 0.0, 1e8 * scale)
    return B + jnp.diag(reg), reg


@partial(jax.jit, static_argnames=("ne", "trafo"))
def preprocess_cholesky(H, S, ne, trafo=True, reg_mask=None):
    A, B, P, _ = proj_out_kernel(H, S, ne)

    A = jax.lax.collapse(jax.lax.collapse(A, 0, 2), 1, 3)
    B = jax.lax.collapse(jax.lax.collapse(B, 0, 2), 1, 3)

    # gather all of it and run cholesky serially
    B, _ = regularize(reshard(B, jax.P()), reg_mask)
    L = jax.lax.linalg.cholesky(B, symmetrize_input=False)
    if not trafo:
        return A, P, L

    I = jnp.eye(L.shape[0], dtype=L.dtype)
    Linv = jax.lax.linalg.triangular_solve(L, I, lower=True)
    C = jnp.einsum("ij,jk,lk->il", Linv, A, Linv.conj())  # Linv @ A @ Linv.T.conj()
    return C, P, Linv.T.conj()


@partial(jax.jit, static_argnames="n")
def postprocess(v, P, *, L=None, Linv=None, n=None):
    if Linv is not None:
        v = jnp.einsum("ij,nj->ni", Linv, v)
    else:
        v = jax.lax.linalg.triangular_solve(L, v.T, left_side=True, lower=True, transpose_a=True, conjugate_a=True).T
    if P is None:
        return v.reshape(v.shape[0], -1, n)
    v = v.reshape(-1, P.shape[0], P.shape[2])
    return jnp.einsum("naj,aij->nai", v, reshard(P, jax.P()))


def preprocess_guess(v, P, L):
    v = jnp.einsum("naj,aji->nai", v, reshard(P, jax.P()).conj())
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


def hermitian_column_view(A):
    spec = jax.typeof(A).sharding.spec
    row_sharded = _uses_mesh_axis(spec, 0)
    column_sharded = _uses_mesh_axis(spec, 1)
    if row_sharded and column_sharded:
        raise ValueError("A cannot shard both axes over mesh axis 'i'")
    return A.T.conj() if row_sharded else A


@jax.jit
def compute_C_row_sh(L, A):
    tmp = _sol1(L, A)
    tmp = reshard(tmp, jax.P("i", None))
    return _sol2(L, tmp)


@jax.jit
def _replicated_matvec(A, x):
    return reshard(A @ x, jax.P())


def precomputed_lanczos_operator(L, A):
    C = compute_C_row_sh(L, hermitian_column_view(A))
    assert jax.typeof(C).sharding.spec == jax.P("i", None)
    return Partial(_replicated_matvec, C)


def flatten(x):
    return jax.lax.collapse(jax.lax.collapse(x, 0, 2), 1, 3)
