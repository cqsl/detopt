import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from functools import partial
from jax.sharding import reshard


from .lanczos import eigsh
from .utils import compute_C_sh, matvec, preprocess_cholesky, preprocess_guess, postprocess
from .nullspace_proj import apply_projl, proj_out_kernel


@partial(jax.jit, static_argnames=("nelec", "k", "precomp", "ncv"))
def generalized_eigh_cholesky_lanczos_tensor(H, S, nelec, k=1, ncv=255, x0=None, key=None, precomp=True):
    ne = nelec
    if isinstance(ne, tuple):
        # TODO figure out a way how to do the projection if the rank of the diagonal blocks is not uniform,
        # we would need to pad, and then index into the matrix to remove the padding, which might be tricky with sharding.
        if ne[0] != ne[1]:
            raise NotImplementedError("use the dense solver for now: --dense_solver")
        ne = ne[0]

    A, P, L = preprocess_cholesky(H, S, ne, trafo=False)
    if x0 is not None:
        assert x0.shape == H.shape[:2]
        x0 = preprocess_guess(x0[None], P, L).ravel()

    if precomp:
        # we can reshard for ~free by taking the hermitian transpose
        # TODO compute A with the desired sharding rightaway
        C = compute_C_sh(L, A.T.conj())
        assert jax.typeof(C).sharding.spec == jax.P(None, "i")
    else:
        C = Partial(matvec, A, L)
    e0, v0, info = eigsh(C, which="SA", k=k, ncv=ncv, dtype=A.dtype, shape=A.shape, v0=x0, key=key)

    e0 = e0[:1]
    v0 = v0.T[:1]
    v0 = postprocess(v0, P, L=L)
    return e0, v0, info[1]


def _concat(Aff, Afl, All):
    Afl = jax.lax.collapse(Afl, 2, 4).reshape(*Aff.shape[:3], -1)
    All = All.reshape(*Afl.shape[2:], *Afl.shape[2:])
    Alf = Afl.transpose(2, 3, 0, 1).conj()
    Afl = reshard(Afl, jax.typeof(Aff).sharding.spec)
    Alf = reshard(Alf, jax.typeof(Aff).sharding.spec)
    All = reshard(All, jax.typeof(Aff).sharding.spec)
    Axf = jnp.concatenate([Aff, Alf], axis=1)
    Axl = jnp.concatenate([Afl, All], axis=1)
    A = jnp.concatenate([Axf, Axl], axis=3)
    return A


def _flatten(x):
    return jax.lax.collapse(jax.lax.collapse(x, 0, 2), 1, 3)


@partial(jax.jit, static_argnames=("nelec", "k", "ncv"))
def generalized_eigh_partial_cholesky_lanczos_tensor(Sff, Hff, Sfl, Hfl, Sll, Hll, nelec, k=1, ncv=255, x0f=None, x0l=None, key=None, precomp=True):
    ne = nelec
    if isinstance(ne, tuple):
        # TODO figure out a way how to do the projection if the rank of the diagonal blocks is not uniform,
        # we would need to pad, and then index into the matrix to remove the padding, which might be tricky with sharding.
        if ne[0] != ne[1]:
            raise NotImplementedError
        ne = ne[0]

    # project out the nullspace
    Aff, Bff, Pl, Pr = proj_out_kernel(Hff, Sff, ne)
    # assemble the full matrices
    A = _flatten(_concat(Aff, apply_projl(Hfl, Pl), Hll))
    B = _flatten(_concat(Bff, apply_projl(Sfl, Pl), Sll))

    # gather all of it
    # TODO write a version using the block-cyclic distributed cholesky
    B = reshard(B, jax.P())
    L = jax.lax.linalg.cholesky(B, symmetrize_input=False)

    if x0l is not None:
        x0f = x0f.reshape(Aff.shape[0], -1)
        x0l = x0l.reshape(Aff.shape[0], -1)
        x0f = jnp.einsum("aj,aji->ai", x0f, Pl.conj())
        x0 = jnp.concatenate([x0f, x0l], axis=1).ravel()
        x0 = jnp.einsum("ji,j->i", L.conj(), x0)
    else:
        x0 = None

    if precomp:
        # we can reshard for ~free by taking the hermitian transpose
        # TODO compute A with the desired sharding rightaway
        C = compute_C_sh(L, A.T.conj())
        assert jax.typeof(C).sharding.spec == jax.P(None, "i")
    else:
        C = Partial(matvec, A, L)

    e0, v0, info = eigsh(C, which="SA", k=k, dtype=A.dtype, shape=A.shape, v0=x0, key=key, ncv=ncv)
    e0 = e0[:1]
    v0 = v0.T[:1]
    v0 = jax.lax.linalg.triangular_solve(L, v0, lower=True, conjugate_a=True)

    v0 = v0.reshape(v0.shape[0], Aff.shape[0], -1)
    v0f = v0[..., : Aff.shape[1]]
    v0l = v0[..., Aff.shape[1] :]
    v0f = jnp.einsum("naj,aij->nai", v0f, Pl)

    return e0[0], v0f[0], v0l[0].reshape(-1, 1), info[1]
