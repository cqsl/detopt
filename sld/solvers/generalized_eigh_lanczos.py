import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from functools import partial
from jax.sharding import reshard

from .lanczos import eigsh
from .utils import matvec, precomputed_lanczos_operator, preprocess_cholesky, preprocess_guess, postprocess, projection_rank, regularize
from .nullspace_proj import _uses_mesh_axis, apply_projl, proj_out_kernel, compute_proj_mat, compute_proj_mat_from_diagonal, apply_proj, sh_diagonal


@partial(jax.jit, static_argnames=("nelec", "k", "precomp", "ncv", "tol"))
def generalized_eigh_cholesky_lanczos_tensor(H, S, nelec, k=1, ncv=255, x0=None, key=None, precomp=True, spin=None, tol=None):
    ne, reg_mask = projection_rank(H.shape[1], nelec, spin)

    A, P, L = preprocess_cholesky(H, S, ne, trafo=False, reg_mask=reg_mask)
    if x0 is not None:
        assert x0.shape == H.shape[:2]
        x0 = preprocess_guess(x0[None], P, L).ravel()

    if precomp:
        C = precomputed_lanczos_operator(L, A)
    else:
        C = Partial(matvec, A, L)
    e0, v0, info = eigsh(C, which="SA", k=k, ncv=ncv, tol=tol, dtype=A.dtype, shape=A.shape, v0=x0, key=key)

    e0 = e0[:1]
    v0 = v0.T[:1]
    v0 = postprocess(v0, P, L=L)
    v0 = reshard(v0, jax.P())
    return e0, v0, info[1]


def _project_excited_effective_matrices(HH, SS, T, G, S_fixed, H_fixed, ne):
    S_inv = jnp.linalg.inv(S_fixed)
    Q = jnp.einsum("ais,st->ait", T, S_inv, optimize="optimal")
    Z = jnp.einsum("ais,ts->ait", T, S_inv.conj(), optimize="optimal")

    S_diagonal = sh_diagonal(SS)
    S_eff_diagonal = S_diagonal - jnp.einsum("ais,ajs->aij", Q, T.conj(), optimize="optimal", out_sharding=jax.typeof(S_diagonal).sharding)
    P = compute_proj_mat_from_diagonal(S_eff_diagonal, ne)
    Pl = reshard(P, jax.P(jax.typeof(HH).sharding.spec[0]))
    Pr = reshard(P, jax.P(jax.typeof(HH).sharding.spec[2]))

    A = apply_proj(HH, Pl, Pr)
    B = apply_proj(SS, Pl, Pr)
    T = apply_projl(T, Pl)
    G = apply_projl(G, Pl)
    Q = apply_projl(Q, Pl)
    Z = apply_projl(Z, Pl)

    A = jax.lax.collapse(jax.lax.collapse(A, 0, 2), 1, 3)
    B = jax.lax.collapse(jax.lax.collapse(B, 0, 2), 1, 3)
    T = jax.lax.collapse(T, 0, 2)
    G = jax.lax.collapse(G, 0, 2)
    Q = jax.lax.collapse(Q, 0, 2)
    Z = jax.lax.collapse(Z, 0, 2)

    A, B, shift, lowrank = _excited_lowrank(A, B, T, G, Q, Z, S_inv, H_fixed)
    return A, B, Pl, shift, lowrank


def _excited_lowrank(A, B, T, G, Q, Z, S_inv, H_fixed):
    row_sharded = _uses_mesh_axis(jax.typeof(A).sharding.spec, 0)
    T_right = reshard(T, jax.P()) if row_sharded else T
    right = jnp.concatenate([Z, G], axis=1)
    right = reshard(right, jax.P()) if row_sharded else right
    B = B + jnp.matmul(-Q, T_right.T.conj(), out_sharding=jax.typeof(B).sharding)
    R = Q @ H_fixed - G
    left = jnp.concatenate([R, -Q], axis=1)
    A = A + jnp.matmul(left, right.T.conj(), out_sharding=jax.typeof(A).sharding)
    shift = jnp.einsum("ij,ji->", S_inv, H_fixed).real
    return A, B, shift, (Q, T_right, left, right)


def _lanczos_exc(A, B, reg_mask, x0, lowrank, k, ncv, tol, key, precomp):
    Q, T_right, left, right = lowrank
    B, reg = regularize(reshard(B, jax.P()), reg_mask)
    L = jax.lax.linalg.cholesky(B, symmetrize_input=False)

    if x0 is not None:
        x0 = jnp.einsum("ji,j->i", L.conj(), x0)

    if precomp:
        C = precomputed_lanczos_operator(L, A)
    else:
        C = Partial(matvec, A, L)
    e0, v0, info = eigsh(C, which="SA", k=k, ncv=ncv, tol=tol, dtype=A.dtype, shape=A.shape, v0=x0, key=key)

    y = v0.T[0]
    # w = L^-† y
    w = jax.lax.linalg.triangular_solve(L, y[:, None], left_side=True, lower=True, transpose_a=True, conjugate_a=True)[:, 0]
    wQ = reshard(Q, jax.P()).T.conj() @ w
    wT = reshard(T_right, jax.P()).T.conj() @ w
    wl = reshard(left, jax.P()).T.conj() @ w
    wr = reshard(right, jax.P()).T.conj() @ w
    s = (y.conj() @ y).real + (wQ.conj() @ wT).real
    if reg is not None:
        s = s - (w.conj() * reg * w).sum().real
    h = (y.conj() @ C(y)).real - (wl.conj() @ wr).real
    return e0[:1], w, info[1], (s, h)


@partial(jax.jit, static_argnames=("nelec", "k", "precomp", "ncv", "tol"))
def generalized_eigh_cholesky_lanczos_exc_tensor(HH, SS, T, G, S_fixed, H_fixed, nelec, k=1, ncv=255, x0=None, key=None, precomp=True, spin=None, tol=None):
    # also returns (v^† SS v, v^† HH v)
    ne, reg_mask = projection_rank(HH.shape[1], nelec, spin)
    A, B, P, shift, lowrank = _project_excited_effective_matrices(HH, SS, T, G, S_fixed, H_fixed, ne)
    P = reshard(P, jax.P())

    if x0 is not None:
        assert x0.shape == HH.shape[:2]
        x0 = jnp.einsum("aj,aji->ai", x0, P.conj()).ravel()

    e0, w, info, sh = _lanczos_exc(A, B, reg_mask, x0, lowrank, k, ncv, tol, key, precomp)
    v0 = jnp.einsum("aj,aij->ai", w.reshape(P.shape[0], P.shape[2]), P)[None]
    return e0 + shift, v0, info, sh


def _interleave(Tf, Tl):
    # same layout as _concat: the 'l' rows are distributed over the 'f' determinants
    Tl = reshard(Tl.reshape(Tf.shape[0], -1, *Tl.shape[1:]), jax.typeof(Tf).sharding.spec)
    return jax.lax.collapse(jnp.concatenate([Tf, Tl], axis=1), 0, 2)


@partial(jax.jit, static_argnames=("nelec", "k", "precomp", "ncv", "tol"))
def generalized_eigh_partial_cholesky_lanczos_exc_tensor(Sff, Hff, Sfl, Hfl, Sll, Hll, Tf, Gf, Tl, Gl, S_fixed, H_fixed, nelec, k=1, ncv=255, x0f=None, x0l=None, key=None, precomp=True, spin=None, tol=None):
    # also returns (v^† S v, v^† H v) of the optimized state
    assert Sll.shape[0] % Sff.shape[0] == 0, "ndet_partial (number of 'f' determinants) must divide the number of 'l' determinants, see _concat"
    nf = Sff.shape[0]
    ne, reg_mask = projection_rank(Hff.shape[1], nelec, spin)
    S_inv = jnp.linalg.inv(S_fixed)
    Qf = jnp.einsum("ais,st->ait", Tf, S_inv, optimize="optimal")
    S_diagonal = sh_diagonal(Sff)
    S_eff_diagonal = S_diagonal - jnp.einsum("ais,ajs->aij", Qf, Tf.conj(), optimize="optimal", out_sharding=jax.typeof(S_diagonal).sharding)
    P = compute_proj_mat_from_diagonal(S_eff_diagonal, ne)
    Pl = reshard(P, jax.P(jax.typeof(Hff).sharding.spec[0]))
    Pr = reshard(P, jax.P(jax.typeof(Hff).sharding.spec[2]))

    A = _flatten(_concat(apply_proj(Hff, Pl, Pr), apply_projl(Hfl, Pl), Hll))
    B = _concat_overlap_replicated(apply_proj(Sff, Pl, Pr), apply_projl(Sfl, Pl), Sll)
    T = _interleave(apply_projl(Tf, Pl), Tl)
    G = _interleave(apply_projl(Gf, Pl), Gl)
    Q = T @ S_inv
    Z = T @ S_inv.T.conj()
    A, B, shift, lowrank = _excited_lowrank(A, B, T, G, Q, Z, S_inv, H_fixed)

    if reg_mask is not None:
        reg_mask = jnp.concatenate([reg_mask.reshape(nf, -1), jnp.ones((nf, Sll.shape[0] // nf), dtype=bool)], axis=1).ravel()

    P = reshard(P, jax.P())
    if x0f is not None:
        x0f = jnp.einsum("aj,aji->ai", x0f, P.conj())
        x0 = jnp.concatenate([x0f, x0l.reshape(nf, -1)], axis=1).ravel()
    else:
        x0 = None

    e0, w, info, sh = _lanczos_exc(A, B, reg_mask, x0, lowrank, k, ncv, tol, key, precomp)
    w = w.reshape(nf, -1)
    v0f = jnp.einsum("aj,aij->ai", w[:, : P.shape[2]], P)
    v0l = w[:, P.shape[2] :].ravel()
    return e0 + shift, v0f, v0l, info, sh


@partial(jax.jit, static_argnames=("nelec"))
def generalized_eigh_cholesky_decomp_tensor(U, S, nelec, spin=None):
    # H is given by U @ U^\dagger

    ne, reg_mask = projection_rank(S.shape[1], nelec, spin)

    P = compute_proj_mat(S, ne)
    # one is gathered, the other one can stay distributed
    Pl = reshard(P, jax.P(jax.typeof(S).sharding.spec[0]))
    Pr = reshard(P, jax.P(jax.typeof(S).sharding.spec[2]))

    S = apply_proj(S, Pl, Pr)
    U = apply_projl(reshard(U, jax.P(jax.typeof(Pr).sharding.spec[0])), Pr)
    U = reshard(U, jax.P())

    # the Cholesky factorization runs on a single device
    S = reshard(S, jax.P())
    S, _ = regularize(jax.lax.collapse(jax.lax.collapse(S, 0, 2), 1, 3), reg_mask)
    L = jax.lax.linalg.cholesky(S, symmetrize_input=False)

    U = jax.lax.collapse(U, 0, 2)
    V = jax.lax.linalg.triangular_solve(L, U, lower=True, left_side=True)

    rank_A = U.shape[-1]
    if rank_A == 1:
        e0 = jnp.einsum("ia,ia->a", V.conj(), V)
        v0 = V.T / jnp.linalg.norm(V)
    else:
        raise NotImplementedError

    v0 = postprocess(v0, Pl, L=L)
    v0 = reshard(v0, jax.P())

    return e0, v0


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


def _concat_overlap_replicated(Bff, Bfl, Bll):
    q = Bff.shape[1]
    Bfl = jax.lax.collapse(Bfl, 2, 4).reshape(*Bff.shape[:3], -1, out_sharding=jax.typeof(Bff).sharding)
    Bll = Bll.reshape(*Bfl.shape[2:], *Bfl.shape[2:])

    Bfx = reshard(jnp.concatenate([Bff, Bfl], axis=3), jax.P())
    Bfl = Bfx[:, :q, :, q:]
    Blx = jnp.concatenate([Bfl.transpose(2, 3, 0, 1).conj(), Bll], axis=3)
    return _flatten(jnp.concatenate([Bfx, Blx], axis=1))


@partial(jax.jit, static_argnames=("nelec", "k", "precomp", "ncv", "tol"))
def generalized_eigh_partial_cholesky_lanczos_tensor(Sff, Hff, Sfl, Hfl, Sll, Hll, nelec, k=1, ncv=255, x0f=None, x0l=None, key=None, precomp=True, spin=None, tol=None):
    assert Sll.shape[0] % Sff.shape[0] == 0, "ndet_partial (number of 'f' determinants) must divide the number of 'l' determinants, see _concat"
    if isinstance(nelec, tuple) and nelec[0] != nelec[1]:
        raise NotImplementedError("partial solver with nelec[0] != nelec[1]")
    ne = nelec[0] if isinstance(nelec, tuple) else nelec

    # project out the nullspace
    Aff, Bff, Pl, Pr = proj_out_kernel(Hff, Sff, ne)
    # assemble the full matrices
    A = _flatten(_concat(Aff, apply_projl(Hfl, Pl), Hll))
    B = _concat_overlap_replicated(Bff, apply_projl(Sfl, Pl), Sll)

    L = jax.lax.linalg.cholesky(B, symmetrize_input=False)

    if x0l is not None:
        x0f = x0f.reshape(Aff.shape[0], -1)
        x0l = x0l.reshape(Aff.shape[0], -1)
        x0f = jnp.einsum("aj,aji->ai", x0f, reshard(Pl, jax.P()).conj())
        x0 = jnp.concatenate([x0f, x0l], axis=1).ravel()
        x0 = jnp.einsum("ji,j->i", L.conj(), x0)
    else:
        x0 = None

    if precomp:
        C = precomputed_lanczos_operator(L, A)
    else:
        C = Partial(matvec, A, L)

    e0, v0, info = eigsh(C, which="SA", k=k, dtype=A.dtype, shape=A.shape, v0=x0, key=key, ncv=ncv, tol=tol)
    e0 = e0[:1]
    v0 = v0.T[:1]
    v0 = jax.lax.linalg.triangular_solve(L, v0, lower=True, conjugate_a=True)

    v0 = v0.reshape(v0.shape[0], Aff.shape[0], -1)
    v0f = v0[..., : Aff.shape[1]]
    v0l = v0[..., Aff.shape[1] :]
    v0f = jnp.einsum("naj,aij->nai", v0f, reshard(Pl, jax.P()))
    v0l = reshard(v0l, jax.P())

    return e0[0], v0f[0], v0l[0].reshape(-1, 1), info[1]
