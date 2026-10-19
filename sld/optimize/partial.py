from functools import partial

import jax
import jax.numpy as jnp
from jax import P
from jax.sharding import reshard

from sld.solvers import generalized_eigh_partial_cholesky_lanczos_tensor
from sld.mels import compute_mels, static_uhf_spin_pattern
from sld.mels._src.hybrid import uhf as hybrid_uhf

from .common import expval_blocks, integrate_and_assemble_all


@partial(jax.jit, static_argnames=("ndet_partial", "batch_size", "batch_size_expval", "batch_size_grad", "batch_size_hess", "cholesky", "lanczos_ncv", "lanczos_tol", "static_spin_blocks", "uhf_backend", "hybrid_gradient_cached_batch_size", "hybrid_gradient_pair_batch_size", "hybrid_hessian_cached_batch_size", "hybrid_hessian_pair_batch_size", "hybrid_factorization", "hybrid_svd_max_degree"), donate_argnames=("S_all", "H_all"))
def optimize_step_partial(
    key,
    X,
    hamiltonian,
    impls,
    ndet_partial=None,
    p_prob=None,
    batch_size=None,
    S_all=None,
    H_all=None,
    cholesky=True,
    batch_size_expval=None,
    batch_size_grad=None,
    batch_size_hess=None,
    lanczos_ncv=255,
    lanczos_tol=None,
    static_spin_blocks=False,
    uhf_backend="pairwise",
    hybrid_gradient_cached_batch_size=1,
    hybrid_gradient_pair_batch_size=128,
    hybrid_hessian_cached_batch_size=16,
    hybrid_hessian_pair_batch_size=128,
    hybrid_svd_rcond=1e-10,
    hybrid_svd_max_degree=4,
    hybrid_factorization="svd",
):
    if isinstance(X.nelec, tuple) and X.nelec[0] != X.nelec[1]:
        raise NotImplementedError("partial optimization with nelec[0] != nelec[1]")
    # optimize only a subset of the determinants' orbitals, lvp for the rest
    # choose randomly of which dets to optimize orbs, and of which only the linear coeff
    # p_prob: (optional) different prob for different dets

    # here we work with determinants where each orbital is assumed to be normalized to norm 1
    # this is done so that we can more easily update S_all and H_all
    # Further we keep the hole state normalized via coeffs (this is ensured automatically by the eigenvalue solver returning a normalized eigenvector)

    if uhf_backend not in ("pairwise", "hybrid", "hybrid_svd"):
        raise ValueError("unknown UHF derivative backend")
    batch_size_expval = batch_size if batch_size_expval is None else batch_size_expval
    batch_size_grad = batch_size if batch_size_grad is None else batch_size_grad
    batch_size_hess = batch_size if batch_size_hess is None else batch_size_hess

    k1, k2, k3 = jax.random.split(key, 3)
    k_full = jnp.sort(jax.random.choice(k1, X.ndet, shape=(ndet_partial,), replace=False, p=p_prob))
    k_lvp = jnp.delete(jnp.arange(X.ndet), k_full, assume_unique_indices=True)

    Xf = X[k_full]
    Xl = X[k_lvp]

    if S_all is None:
        assert H_all is None
        # normalize all determinants to orbital norm 1
        Xl = Xl.normalize_orbs()
        Sll, Hll = expval_blocks(impls, Xl, hamiltonian, batch_size_expval, uhf_backend, hybrid_svd_rcond, min(hybrid_svd_max_degree, 2), hybrid_factorization, hybrid_hessian_cached_batch_size, hybrid_hessian_pair_batch_size)
        Sll = reshard(Sll, P())
        Hll = reshard(Hll, P())
    else:
        Sll = jnp.take_along_axis(jnp.take_along_axis(S_all, k_lvp[None, :], 1), k_lvp[:, None], 0)[:, None, :, None]
        Hll = jnp.take_along_axis(jnp.take_along_axis(H_all, k_lvp[None, :], 1), k_lvp[:, None], 0)[:, None, :, None]

    # we perform a random rotation mixing all the orbitals, so it is sufficient to optimize the first one.
    Xf = Xf.choose_random_orbitals_to_optimize(k2, optimize_first_only=True)
    if static_spin_blocks or uhf_backend != "pairwise":
        Xf = Xf.replace(s=static_uhf_spin_pattern(Xf.ndet, nshards=1 if uhf_backend in ("hybrid", "hybrid_svd") else None))
    # orthogonalize the orbitals; in particular this normalizes them (which might already be sufficient to ensure numerical stability)
    Xf = Xf.orthogonalize_orbitals()
    # orthogonalization and random unitary rotation are an element of the SL group
    Xf = Xf.randomly_rotate_orbitals(k3)

    if uhf_backend in ("hybrid", "hybrid_svd"):
        if jax.device_count() == 1:
            Sff, Hff = hybrid_uhf.hessian(Xf, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 4), cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
            Sfl, Hfl = hybrid_uhf.gradient(Xf, Xl, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 3), cached_batch_size=hybrid_gradient_cached_batch_size, pair_batch_size=hybrid_gradient_pair_batch_size)
        else:
            Xf_bra = reshard(Xf, P("i"))
            Sff, Hff = hybrid_uhf.hessian_sharded(Xf_bra, arrays=hamiltonian, Y=Xf, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 4), cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
            Sfl, Hfl = hybrid_uhf.gradient_sharded(Xf_bra, Xl, arrays=hamiltonian, rcond=hybrid_svd_rcond, factorization=hybrid_factorization, max_degree=min(hybrid_svd_max_degree, 3), cached_batch_size=hybrid_gradient_cached_batch_size, pair_batch_size=hybrid_gradient_pair_batch_size)
    else:
        Sff, Hff, *_ = compute_mels(impls.hess, Xf, reshard(Xf, P("i")), arrays=hamiltonian, batch_size=batch_size_hess)
        Sfl, Hfl, *_ = compute_mels(impls.grad, Xf, reshard(Xl, P("i")), arrays=hamiltonian, batch_size=batch_size_grad)

    x0f = Xf.get_orbitals_weighted()
    x0l = Xl.coeffs
    if cholesky:
        e0, v0f, v0l, info = generalized_eigh_partial_cholesky_lanczos_tensor(Sff, Hff, Sfl, Hfl, Sll, Hll, nelec=Xf.nelec, ncv=lanczos_ncv, tol=lanczos_tol, x0f=x0f, x0l=x0l)
    else:
        raise NotImplementedError("--dense_solver")
    coeffs_l = v0l.ravel()
    coeffs_f = jnp.linalg.norm(v0f, axis=1)
    v0f = v0f / coeffs_f[:, None]

    Xl = Xl.replace(coeffs=coeffs_l)

    Xf = Xf.set_orbitals(v0f)
    Xf = Xf.replace(coeffs=coeffs_f)
    Xf = Xf.clear_orbitals_to_optimize()

    # concatenate
    X = Xf + Xl
    Sff = jnp.einsum("aibj,ai,bj->ab", Sff, reshard(v0f, P(jax.typeof(Sff).sharding.spec[0])).conj(), reshard(v0f, P(jax.typeof(Sff).sharding.spec[2])), out_sharding=P())
    Hff = jnp.einsum("aibj,ai,bj->ab", Hff, reshard(v0f, P(jax.typeof(Hff).sharding.spec[0])).conj(), reshard(v0f, P(jax.typeof(Hff).sharding.spec[2])), out_sharding=P())
    S_all, H_all = integrate_and_assemble_all(v0f, Sff, Hff, Sfl, Hfl, Sll, Hll)

    return X, e0, info, S_all, H_all
