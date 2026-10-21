import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import P
from jax.sharding import reshard

from sld.mels import compute_grad_weighted, compute_mels, static_uhf_spin_pattern
from sld.mels._src.hybrid import uhf as hybrid_uhf
from sld.solvers import generalized_eigh_partial_cholesky_lanczos_exc_tensor

from .common import expval_blocks, integrate_and_assemble_all
from .exc import compute_s_h_all, uniform_spin_count_pattern


@partial(
    jax.jit,
    static_argnames=("nper", "ndet_partial", "batch_size", "batch_size_expval", "batch_size_grad", "batch_size_hess", "keep_order", "lanczos_ncv", "lanczos_tol", "static_spin_blocks", "uniform_spin_count", "uhf_backend", "hybrid_gradient_cached_batch_size", "hybrid_gradient_pair_batch_size", "hybrid_hessian_cached_batch_size", "hybrid_hessian_pair_batch_size", "hybrid_factorization", "hybrid_svd_max_degree"),
    donate_argnames=("S_all", "H_all"),
)
def optimize_step_exc_partial(
    key,
    X,
    hamiltonian,
    impls,
    nper=None,
    ndet_partial=None,
    p_prob=None,
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
    # like optimize_step_exc, but only the orbitals of ndet_partial of the determinants of the optimized state are optimized, lvp for the rest
    # p_prob: (optional) (nstates, nper) probabilities to choose the determinants of each state for orbital optimization
    assert nper is not None
    assert ndet_partial is not None and ndet_partial < nper
    if uhf_backend not in ("pairwise", "hybrid", "hybrid_svd"):
        raise ValueError("unknown UHF derivative backend")
    if uniform_spin_count and (static_spin_blocks or uhf_backend != "pairwise"):
        raise ValueError("uniform_spin_count requires the pairwise backend and dynamic spin blocks")
    hybrid = uhf_backend in ("hybrid", "hybrid_svd")
    batch_size_expval = batch_size if batch_size_expval is None else batch_size_expval
    batch_size_grad = batch_size if batch_size_grad is None else batch_size_grad
    batch_size_hess = batch_size if batch_size_hess is None else batch_size_hess
    ndet = X.ndet
    nstates = ndet // nper

    k1, k2, k3, k4 = jax.random.split(key, 4)
    if j_opt is None:
        j_opt = jax.random.randint(k1, shape=(), minval=0, maxval=nstates, dtype=jnp.int32)

    k_f = jax.random.choice(k4, nper, shape=(ndet_partial,), replace=False, p=None if p_prob is None else p_prob[j_opt]).astype(jnp.int32)
    if static_spin_blocks or uhf_backend != "pairwise":
        k_f = jax.random.permutation(jax.random.fold_in(k4, 1), k_f)
    k_l = jnp.delete(jnp.arange(nper, dtype=jnp.int32), k_f, assume_unique_indices=True)
    k_opt = (j_opt * nper + jnp.concatenate([k_f, k_l])).astype(jnp.int32)
    k_fix = jnp.delete(jnp.arange(ndet, dtype=jnp.int32), j_opt * nper + jnp.arange(nper, dtype=jnp.int32), assume_unique_indices=True)

    Xf = X[k_opt[:ndet_partial]]
    Xl = X[k_opt[ndet_partial:]]
    X_fix = X[k_fix]

    if S_all is None:
        assert H_all is None
        X_fix, S, H = compute_s_h_all(impls, X_fix, hamiltonian, nstates - 1, batch_size_expval, backend=uhf_backend, rcond=hybrid_svd_rcond, max_degree=min(hybrid_svd_max_degree, 2), factorization=hybrid_factorization, cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
        Xl = Xl.normalize_orbs()
    else:
        S = jnp.delete(jnp.delete(S_all, j_opt, axis=1, assume_unique_indices=True), j_opt, axis=0, assume_unique_indices=True)
        H = jnp.delete(jnp.delete(H_all, j_opt, axis=1, assume_unique_indices=True), j_opt, axis=0, assume_unique_indices=True)

    Xf = Xf.choose_random_orbitals_to_optimize(k2, optimize_first_only=False)
    if uniform_spin_count:
        Xf = Xf.replace(s=uniform_spin_count_pattern(jax.random.fold_in(k2, 731), Xf.ndet))
    if static_spin_blocks or uhf_backend != "pairwise":
        Xf = Xf.replace(s=static_uhf_spin_pattern(Xf.ndet, nshards=1 if hybrid else None))
    Xf = Xf.orthogonalize_orbitals()
    Xf = Xf.randomly_rotate_orbitals(k3)

    # lvp determinants vs themselves and the fixed states
    nl = Xl.ndet
    Slr, Hlr = expval_blocks(impls, Xl, hamiltonian, batch_size_expval, uhf_backend, hybrid_svd_rcond, min(hybrid_svd_max_degree, 2), hybrid_factorization, hybrid_hessian_cached_batch_size, hybrid_hessian_pair_batch_size, Y=Xl + X_fix)
    Slr = reshard(Slr, P())
    Hlr = reshard(Hlr, P())
    Sll, Hll = Slr[:, :, :nl], Hlr[:, :, :nl]
    c_fix = X_fix.coeffs.reshape(nstates - 1, nper)
    Tl = jnp.einsum("lsk,sk->ls", Slr[:, 0, nl:, 0].reshape(nl, nstates - 1, nper), c_fix)
    Gl = jnp.einsum("lsk,sk->ls", Hlr[:, 0, nl:, 0].reshape(nl, nstates - 1, nper), c_fix)

    # shard over the determinants of each state
    X_fix_ = jax.tree.map(lambda x: x.reshape(nstates - 1, nper, math.prod(x.shape[1:])).transpose(1, 0, 2).reshape(x.shape), X_fix)
    fixed_state = jnp.tile(jnp.arange(nstates - 1, dtype=jnp.int32), nper)
    ket_weights = X_fix_.coeffs[:, None] * jax.nn.one_hot(fixed_state, nstates - 1, dtype=X_fix_.coeffs.dtype)
    if hybrid:
        if jax.device_count() == 1:
            Sff, Hff = hybrid_uhf.hessian(Xf, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 4), cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
            Sfl, Hfl = hybrid_uhf.gradient(Xf, Xl, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 3), cached_batch_size=hybrid_gradient_cached_batch_size, pair_batch_size=hybrid_gradient_pair_batch_size)
            T, G = hybrid_uhf.gradient_weighted(Xf, X_fix_, ket_weights, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 3), cached_batch_size=hybrid_gradient_cached_batch_size, pair_batch_size=hybrid_gradient_pair_batch_size)
        else:
            Xf_bra = reshard(Xf, P("i"))
            Sff, Hff = hybrid_uhf.hessian_sharded(Xf_bra, arrays=hamiltonian, Y=Xf, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 4), cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
            Sfl, Hfl = hybrid_uhf.gradient_sharded(Xf_bra, Xl, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 3), cached_batch_size=hybrid_gradient_cached_batch_size, pair_batch_size=hybrid_gradient_pair_batch_size)
            T, G = hybrid_uhf.gradient_weighted_sharded(Xf_bra, X_fix_, ket_weights, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 3), cached_batch_size=hybrid_gradient_cached_batch_size, pair_batch_size=hybrid_gradient_pair_batch_size)
    else:
        Sff, Hff, *_ = compute_mels(impls.hess, Xf, reshard(Xf, P("i")), arrays=hamiltonian, batch_size=batch_size_hess)
        Sfl, Hfl, *_ = compute_mels(impls.grad, Xf, reshard(Xl, P("i")), arrays=hamiltonian, batch_size=batch_size_grad)
        T, G = compute_grad_weighted(impls.grad, Xf, reshard(X_fix_, P("i")), reshard(ket_weights, P("i")), arrays=hamiltonian, batch_size=batch_size_grad)

    if cholesky:
        e, v0f, v0l, info, (normsq_opt, h_opt) = generalized_eigh_partial_cholesky_lanczos_exc_tensor(Sff, Hff, Sfl, Hfl, Sll, Hll, T, G, Tl, Gl, S, H, nelec=Xf.nelec, ncv=lanczos_ncv, tol=lanczos_tol, x0f=Xf.get_orbitals_weighted(), x0l=Xl.coeffs, spin=getattr(Xf, "s", None))
    else:
        raise NotImplementedError("--dense_solver")
    e0 = e[0]

    # normalize the new state
    v0f_normalized = v0f * jax.lax.rsqrt(normsq_opt)
    coeffs_l = v0l * jax.lax.rsqrt(normsq_opt)
    # normalize the orbitals
    coeffs_f = jnp.linalg.norm(v0f_normalized, axis=1)
    nonzero = coeffs_f > 0
    safe_norm = jnp.where(nonzero, coeffs_f, 1)
    v0f = jnp.where(nonzero[:, None], v0f_normalized / safe_norm[:, None], Xf.get_orbitals())

    Xf = Xf.set_orbitals(v0f).replace(coeffs=coeffs_f).clear_orbitals_to_optimize()
    Xl = Xl.replace(coeffs=coeffs_l)

    # concatenate
    X = Xf + Xl + X_fix
    T = jnp.einsum("ais,ai->s", reshard(T, P()), v0f_normalized.conj()) + jnp.einsum("ls,l->s", Tl, coeffs_l.conj())
    G = jnp.einsum("ais,ai->s", reshard(G, P()), v0f_normalized.conj()) + jnp.einsum("ls,l->s", Gl, coeffs_l.conj())
    S_all, H_all = integrate_and_assemble_all(jnp.ones((1, 1), dtype=T.dtype), 1.0, h_opt / normsq_opt, T[None, None, :, None], G[None, None, :, None], S[:, None, :, None], H[:, None, :, None], sum_f=True)

    if keep_order:
        # restore original ordering
        perm = jnp.argsort(jnp.concatenate([k_opt, k_fix]))
        X = X[perm]
        j_fix = jnp.delete(jnp.arange(nstates, dtype=jnp.int32), j_opt, assume_unique_indices=True)
        state_perm = jnp.argsort(jnp.concatenate([jnp.atleast_1d(j_opt), j_fix]))
        S_all = S_all[state_perm, :][:, state_perm]
        H_all = H_all[state_perm, :][:, state_perm]

    return X, e0, info, S_all, H_all
