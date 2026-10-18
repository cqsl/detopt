import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import reshard


def apply_proj(S, Pl, Pr):
    return jnp.einsum("aij,aibk,bkl->ajbl", Pl.conj(), S, Pr)


def apply_projl(T, Pl):
    return jnp.einsum("aij,ai...->aj...", Pl.conj(), T)


def apply_projr(T, Pr):
    return jnp.einsum("...bk,bkl->...bl", T, Pr)


@partial(jax.jit, static_argnums=1)
def _compute_proj_mat_num(S_diag, r):
    # NB eigvals are all the same if the dets are normalized, eigvecs are different though
    s, v = jnp.linalg.eigh(S_diag)
    s = s[:, r:]
    v = v[:, :, r:]
    return v


def _uses_mesh_axis(spec, array_axis, mesh_axis="i"):
    if array_axis >= len(spec):
        return False
    partition = spec[array_axis]
    return partition == mesh_axis or (
        isinstance(partition, tuple) and mesh_axis in partition
    )


@jax.jit
@partial(jax.smap, axis_name="i", out_axes=0)
def _sh_diagonal_columns(S):
    n, _, m, _ = S.shape
    j = jnp.arange(m, dtype=jnp.int32)
    i = j + jax.lax.axis_index("i") * m
    return S[i, :, j, :]


@jax.jit
@partial(jax.smap, axis_name="i", out_axes=0)
def _sh_diagonal_rows(S):
    n, _, _, _ = S.shape
    i = jnp.arange(n, dtype=jnp.int32)
    j = i + jax.lax.axis_index("i") * n
    return S[i, :, j, :]


@jax.jit
def sh_diagonal(S):
    spec = jax.typeof(S).sharding.spec
    row_sharded = _uses_mesh_axis(spec, 0)
    column_sharded = _uses_mesh_axis(spec, 2)
    if row_sharded and column_sharded:
        raise ValueError("S cannot shard both determinant axes over mesh axis 'i'")
    if row_sharded:
        return _sh_diagonal_rows(S)
    if column_sharded:
        return _sh_diagonal_columns(S)
    indices = jnp.arange(S.shape[0], dtype=jnp.int32)
    return S[indices, :, indices, :]


@partial(jax.jit, static_argnums=1)
def compute_proj_mat(S, ne):
    S_diag = sh_diagonal(S)
    return compute_proj_mat_from_diagonal(S_diag, ne)


@partial(jax.jit, static_argnums=1)
def compute_proj_mat_from_diagonal(S_diag, ne):
    return _compute_proj_mat_num(S_diag, ne - 1)


@partial(jax.jit, static_argnums=2)
def proj_out_kernel(H, S, ne):
    P = compute_proj_mat(S, ne)
    # one is gathered, the other one can stay distributed
    Pl = reshard(P, jax.P(jax.typeof(H).sharding.spec[0]))
    Pr = reshard(P, jax.P(jax.typeof(H).sharding.spec[2]))
    return apply_proj(H, Pl, Pr), apply_proj(S, Pl, Pr), Pl, Pr
