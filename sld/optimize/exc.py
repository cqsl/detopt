import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import P
from jax.sharding import reshard

from sld.mels import compute_grad_weighted, compute_mels, static_uhf_spin_pattern
from sld.mels._src.hybrid import uhf as hybrid_uhf
from sld.solvers import generalized_eigh_cholesky_lanczos_exc_tensor

from .common import expval_blocks, integrate_and_assemble_all


@jax.jit
def _diagonal(S):
    n, _, m, _ = S.shape
    i = j = jnp.arange(m, dtype=jnp.int32)
    return S[i, :, j, :]


def uniform_spin_count_pattern(key, ndet):
    count_key, shuffle_key = jax.random.split(key)
    nup = jax.random.randint(count_key, (), 0, ndet + 1)
    return jax.random.permutation(shuffle_key, (jnp.arange(ndet) >= nup).astype(jnp.int32))


@partial(jax.jit, static_argnames=('batch_size', 'nstates', 'backend', 'max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def compute_s_h_all(impls, X, hamiltonian, nstates, batch_size=None, *, backend="pairwise", rcond=1e-10, max_degree=2, factorization="svd", cached_batch_size=16, pair_batch_size=128):
    nper = X.ndet // nstates
    # normalize all determinants to orbital norm 1
    X_fix = X.normalize_orbs()
    S_fix_fix, H_fix_fix = expval_blocks(impls, X_fix, hamiltonian, batch_size, backend, rcond, max_degree, factorization, cached_batch_size, pair_batch_size)
    # gather and compute the rest serially, since nstates need not be divisible by the number of devices
    S_fix_fix = reshard(S_fix_fix, P())
    H_fix_fix = reshard(H_fix_fix, P())

    S_fix_fix = jax.lax.collapse(jax.lax.collapse(S_fix_fix, 0, 2), 1, 3).reshape(nstates, nper, -1).reshape(nstates, nper, nstates, nper)
    H_fix_fix = jax.lax.collapse(jax.lax.collapse(H_fix_fix, 0, 2), 1, 3).reshape(nstates, nper, -1).reshape(nstates, nper, nstates, nper)
    # pre-normalize the vectors
    c = X_fix.coeffs.reshape(nstates, nper)
    c = c * jax.lax.rsqrt((c * c.conj()).sum(axis=1, keepdims=True))
    # normalize states
    D = _diagonal(S_fix_fix)
    norms_states = reshard(jnp.einsum("ai,aij,aj->a", c.conj(), D, c), P())
    c = c * jax.lax.rsqrt(norms_states)[:, None]
    X_fix = X_fix.replace(coeffs=c.ravel())
    # sum over states
    S = jnp.einsum("ai,aibj,bj->ab", c.conj(), S_fix_fix, c)
    H = jnp.einsum("ai,aibj,bj->ab", c.conj(), H_fix_fix, c)
    return X_fix, reshard(S, P()), reshard(H, P())

@partial(
    jax.jit,
    static_argnames=("nper", "batch_size", "batch_size_expval", "batch_size_grad", "batch_size_hess", "keep_order", "lanczos_ncv", "lanczos_tol", "static_spin_blocks", "uniform_spin_count", "uhf_backend", "hybrid_gradient_cached_batch_size", "hybrid_gradient_pair_batch_size", "hybrid_hessian_cached_batch_size", "hybrid_hessian_pair_batch_size", "hybrid_factorization", "hybrid_svd_max_degree"),
    donate_argnames=("S_all", "H_all"),
)
def optimize_step_exc(
    key,
    X,
    hamiltonian,
    impls,
    nper=None,
    batch_size=None,
    S_all=None,
    H_all=None,
    keep_order=False,
    j_opt=None,
    cholesky=True,
    batch_size_expval=None,
    batch_size_grad=None,
    batch_size_hess=None,
    lanczos_ncv=255,
    lanczos_tol=None,
    static_spin_blocks=False,
    uniform_spin_count=False,
    uhf_backend="pairwise",
    hybrid_gradient_cached_batch_size=1,
    hybrid_gradient_pair_batch_size=128,
    hybrid_hessian_cached_batch_size=16,
    hybrid_hessian_pair_batch_size=128,
    hybrid_svd_rcond=1e-10,
    hybrid_svd_max_degree=4,
    hybrid_factorization="svd",
):
    assert nper is not None
    if uhf_backend not in ("pairwise", "hybrid", "hybrid_svd"):
        raise ValueError("unknown UHF derivative backend")
    batch_size_expval = batch_size if batch_size_expval is None else batch_size_expval
    batch_size_grad = batch_size if batch_size_grad is None else batch_size_grad
    batch_size_hess = batch_size if batch_size_hess is None else batch_size_hess
    ndet = X.ndet
    nstates = ndet // nper

    k1, k2, k3 = jax.random.split(key, 3)
    if j_opt is None:
        j_opt = jax.random.randint(k1, shape=(), minval=0, maxval=nstates, dtype=jnp.int32)

    k_opt = (j_opt * nper + jnp.arange(nper, dtype=jnp.int32)).astype(jnp.int32)
    k_fix = jnp.delete(jnp.arange(ndet, dtype=jnp.int32), k_opt, assume_unique_indices=True)

    X_opt = X[k_opt]
    X_fix = X[k_fix]

    if S_all is None:
        assert H_all is None
        # X_fix holds the other nstates - 1 states
        X_fix, S, H = compute_s_h_all(impls, X_fix, hamiltonian, nstates - 1, batch_size_expval, backend=uhf_backend, rcond=hybrid_svd_rcond, max_degree=min(hybrid_svd_max_degree, 2), factorization=hybrid_factorization, cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
    else:
        S = jnp.delete(jnp.delete(S_all, j_opt, axis=1, assume_unique_indices=True), j_opt, axis=0, assume_unique_indices=True)
        H = jnp.delete(jnp.delete(H_all, j_opt, axis=1, assume_unique_indices=True), j_opt, axis=0, assume_unique_indices=True)

    # we perform a random rotation mixing all the orbitals, so it is sufficient to optimize the first one.
    if static_spin_blocks or uhf_backend != "pairwise":
        k_shuffle, k2 = jax.random.split(k2)
        X_opt = X_opt[jax.random.permutation(k_shuffle, X_opt.ndet)]
    if uniform_spin_count and (static_spin_blocks or uhf_backend != "pairwise"):
        raise ValueError("uniform_spin_count requires the pairwise backend and dynamic spin blocks")
    X_opt = X_opt.choose_random_orbitals_to_optimize(k2, optimize_first_only=False)
    if uniform_spin_count:
        X_opt = X_opt.replace(s=uniform_spin_count_pattern(jax.random.fold_in(k2, 731), X_opt.ndet))
    if static_spin_blocks or uhf_backend != "pairwise":
        X_opt = X_opt.replace(s=static_uhf_spin_pattern(X_opt.ndet, nshards=1 if uhf_backend in ("hybrid", "hybrid_svd") else None))
    X_opt = X_opt.orthogonalize_orbitals()
    X_opt = X_opt.randomly_rotate_orbitals(k3)

    if uhf_backend in ("hybrid", "hybrid_svd"):
        if jax.device_count() == 1:
            SS, HH = hybrid_uhf.hessian(X_opt, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 4), cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
        else:
            X_opt_bra = reshard(X_opt, P("i"))
            SS, HH = hybrid_uhf.hessian_sharded(X_opt_bra, arrays=hamiltonian, Y=X_opt, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 4), cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
    else:
        SS, HH, *_ = compute_mels(impls.hess, X_opt, reshard(X_opt, P("i")), arrays=hamiltonian, batch_size=batch_size_hess)

    # shard over the determinants of each state
    X_fix_ = jax.tree.map(lambda x: x.reshape(nstates - 1, nper, math.prod(x.shape[1:])).transpose(1, 0, 2).reshape(x.shape), X_fix)
    fixed_state = jnp.tile(jnp.arange(nstates - 1, dtype=jnp.int32), nper)
    ket_weights = X_fix_.coeffs[:, None] * jax.nn.one_hot(fixed_state, nstates - 1, dtype=X_fix_.coeffs.dtype)
    if uhf_backend in ("hybrid", "hybrid_svd"):
        if jax.device_count() == 1:
            T, G = hybrid_uhf.gradient_weighted(X_opt, X_fix_, ket_weights, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 3), cached_batch_size=hybrid_gradient_cached_batch_size, pair_batch_size=hybrid_gradient_pair_batch_size)
        else:
            T, G = hybrid_uhf.gradient_weighted_sharded(X_opt_bra, X_fix_, ket_weights, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 3), cached_batch_size=hybrid_gradient_cached_batch_size, pair_batch_size=hybrid_gradient_pair_batch_size)
    else:
        T, G = compute_grad_weighted(impls.grad, X_opt, reshard(X_fix_, P("i")), reshard(ket_weights, P("i")), arrays=hamiltonian, batch_size=batch_size_grad)

    if cholesky:
        x0_opt = X_opt.get_orbitals_weighted()
        e, v_opt, info, (normsq_opt, h_opt) = generalized_eigh_cholesky_lanczos_exc_tensor(HH, SS, T, G, S, H, nelec=X_opt.nelec, ncv=lanczos_ncv, tol=lanczos_tol, x0=x0_opt, spin=getattr(X_opt, "s", None))
    else:
        raise NotImplementedError("--dense_solver")

    e0 = e[0]
    v0_opt = v_opt[0]

    # normalize the new state
    v0_opt_normalized = v0_opt * jax.lax.rsqrt(normsq_opt)
    # normalize the orbitals
    coeffs_opt = jnp.linalg.norm(v0_opt_normalized, axis=1)
    nonzero = coeffs_opt > 0
    safe_norm = jnp.where(nonzero, coeffs_opt, 1)
    v0_opt = jnp.where(nonzero[:, None], v0_opt_normalized / safe_norm[:, None], X_opt.get_orbitals())

    X_opt = X_opt.set_orbitals(v0_opt)
    X_opt = X_opt.replace(coeffs=coeffs_opt)
    X_opt = X_opt.clear_orbitals_to_optimize()

    # concatenate
    X = X_opt + X_fix
    S_all, H_all = integrate_and_assemble_all(v0_opt_normalized, 1.0, h_opt / normsq_opt, T[..., None], G[..., None], S[:, None, :, None], H[:, None, :, None], sum_f=True)

    if keep_order:
        # restore original ordering
        perm = jnp.argsort(jnp.concatenate([k_opt, k_fix]))
        X = X[perm]
        # S_all and H_all are indexed per state, so they need a state-level permutation
        j_fix = jnp.delete(jnp.arange(nstates, dtype=jnp.int32), j_opt, assume_unique_indices=True)
        state_perm = jnp.argsort(jnp.concatenate([jnp.atleast_1d(j_opt), j_fix]))
        S_all = S_all[state_perm, :][:, state_perm]
        H_all = H_all[state_perm, :][:, state_perm]

    return X, e0, info, S_all, H_all
