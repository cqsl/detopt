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


# TODO maybe write a version which does it analytically, taking the other orbitals as input


@partial(jax.jit, static_argnums=1)
def _compute_proj_mat_num(S_diag, r):
    # NB eigvals are all the same if the dets are normalized, eigvecs are different though
    s, v = jnp.linalg.eigh(S_diag)
    s = s[:, r:]
    v = v[:, :, r:]
    return v


@jax.jit
@partial(jax.smap, axis_name="i", out_axes=0)
def sh_diagonal(S):
    # assumes S is sharded as P(None, None, 'i', None)
    n, _, m, _ = S.shape

    j = jnp.arange(m, dtype=jnp.int32)
    i = j + jax.lax.axis_index("i") * m

    return S[i, :, j, :]


@partial(jax.jit, static_argnums=1)
def compute_proj_mat(S, ne):
    r = ne - 1
    S_diag = sh_diagonal(S)
    return _compute_proj_mat_num(S_diag, r)


@partial(jax.jit, static_argnums=2)
def proj_out_kernel(H, S, ne):
    P = compute_proj_mat(S, ne)
    # one is gathered, the other one can stay distributed
    Pl = reshard(P, jax.P(jax.typeof(H).sharding.spec[0]))
    Pr = reshard(P, jax.P(jax.typeof(H).sharding.spec[2]))
    return apply_proj(H, Pl, Pr), apply_proj(S, Pl, Pr), Pl, Pr
