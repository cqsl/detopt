from functools import partial

import jax
import jax.numpy as jnp
from jax import P
from jax.sharding import reshard

from sld.solvers import generalized_eigh_cholesky_lanczos_tensor, generalized_eigh_lowrank_tensor

from sld.mels import compute_mels, compute_mels_static_uhf_spin_blocks, static_uhf_spin_pattern
from sld.mels._src.hybrid import uhf as hybrid_uhf


def _prepare_step(key, X, static_spin_blocks=False, uhf_backend="pairwise", nshards=1):
    if static_spin_blocks or uhf_backend != "pairwise":
        if not isinstance(X.nelec, tuple):
            raise ValueError("static spin blocks require UHF determinants")
        k_permutation, k_spin, k_rotation = jax.random.split(key, 3)
        X = X[jax.random.permutation(k_permutation, X.ndet)]
        X = X.choose_random_orbitals_to_optimize(k_spin, optimize_first_only=True)
        X = X.replace(s=static_uhf_spin_pattern(X.ndet, nshards=1 if uhf_backend in ("hybrid", "hybrid_svd") else nshards))
    else:
        k_spin, k_rotation = jax.random.split(key)
        X = X.choose_random_orbitals_to_optimize(k_spin, optimize_first_only=True)
    X = X.orthogonalize_orbitals()
    return X.randomly_rotate_orbitals(k_rotation)


def _compute_matrix_elements(
    X,
    hamiltonian,
    impls,
    batch_size=None,
    static_spin_blocks=False,
    uhf_backend="pairwise",
    hybrid_hessian_cached_batch_size=16,
    hybrid_hessian_pair_batch_size=128,
    hybrid_svd_rcond=1e-10,
    hybrid_svd_max_degree=4,
    hybrid_factorization="svd",
    nshards=1,
):
    if static_spin_blocks or uhf_backend != "pairwise":
        if not isinstance(X.nelec, tuple) or impls.hess_spin_blocks is None:
            raise ValueError("static spin blocks require unprojected UHF matrix elements")
    if uhf_backend in ("hybrid", "hybrid_svd"):
        if nshards == 1:
            return hybrid_uhf.hessian(X, arrays=hamiltonian, rcond=hybrid_svd_rcond, max_degree=min(hybrid_svd_max_degree, 4), factorization=hybrid_factorization, cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
        return hybrid_uhf.hessian_sharded(reshard(X, P("i")), arrays=hamiltonian, Y=X, rcond=hybrid_svd_rcond, max_degree=min(hybrid_svd_max_degree, 4), factorization=hybrid_factorization, cached_batch_size=hybrid_hessian_cached_batch_size, pair_batch_size=hybrid_hessian_pair_batch_size)
    if static_spin_blocks:
        return compute_mels_static_uhf_spin_blocks(impls.hess_spin_blocks, X, arrays=hamiltonian, batch_size=batch_size, nshards=nshards)
    return compute_mels(impls.hess, X, reshard(X, P("i")), arrays=hamiltonian, batch_size=batch_size)


def _solve(X, H, S, cholesky=True, lanczos_ncv=255, lanczos_tol=None):
    if cholesky:
        x0 = X.get_orbitals_weighted()
        e, v, info = generalized_eigh_cholesky_lanczos_tensor(H, S, nelec=X.nelec, ncv=lanczos_ncv, tol=lanczos_tol, x0=x0, spin=getattr(X, "s", None))
    else:
        H = reshard(H, jax.P())
        S = reshard(S, jax.P())
        if isinstance(X.nelec, tuple) and X.nelec[0] != X.nelec[1]:
            n_zero_eigvals = 0
        else:
            n_zero_eigvals = int(X.n_zero_eigvals)
        e, v = generalized_eigh_lowrank_tensor(H, S, n_zero_eigvals=n_zero_eigvals, rcond=1e-13)
        info = None
    return e, v, info


def _update_step(X, e, v, info):
    e0 = e[0]
    v0 = v[0]
    coeffs = jnp.linalg.norm(v0, axis=1)
    v0 = v0 / coeffs[:, None]
    X = X.set_orbitals(v0).replace(coeffs=coeffs).clear_orbitals_to_optimize()
    return X, e0, info


@partial(jax.jit, static_argnames=("batch_size", "cholesky", "lanczos_ncv", "lanczos_tol", "static_spin_blocks", "uhf_backend", "hybrid_hessian_cached_batch_size", "hybrid_hessian_pair_batch_size", "hybrid_factorization", "hybrid_svd_max_degree", "nshards"))
def _optimize_step(
    key,
    X,
    hamiltonian,
    impls,
    batch_size=None,
    cholesky=True,
    lanczos_ncv=255,
    lanczos_tol=None,
    static_spin_blocks=False,
    uhf_backend="pairwise",
    hybrid_hessian_cached_batch_size=16,
    hybrid_hessian_pair_batch_size=128,
    hybrid_svd_rcond=1e-10,
    hybrid_svd_max_degree=4,
    hybrid_factorization="svd",
    nshards=1,
):
    # cholesky: use cholesky+lanczos to solve the eigenvalue problem if True; use dense solver if False
    X = _prepare_step(key, X, static_spin_blocks, uhf_backend, nshards)
    S, H = _compute_matrix_elements(X, hamiltonian, impls, batch_size, static_spin_blocks, uhf_backend, hybrid_hessian_cached_batch_size, hybrid_hessian_pair_batch_size, hybrid_svd_rcond, hybrid_svd_max_degree, hybrid_factorization, nshards)
    e, v, info = _solve(X, H, S, cholesky, lanczos_ncv, lanczos_tol)
    return _update_step(X, e, v, info)


def optimize_step(key, X, hamiltonian, impls, batch_size=None, cholesky=True, lanczos_ncv=255, lanczos_tol=None, static_spin_blocks=False, uhf_backend=None, hybrid_hessian_cached_batch_size=16, hybrid_hessian_pair_batch_size=128, hybrid_svd_rcond=1e-10, hybrid_svd_max_degree=4, hybrid_factorization="svd"):
    uhf_backend = "pairwise" if uhf_backend is None else uhf_backend
    if uhf_backend not in ("pairwise", "hybrid", "hybrid_svd"):
        raise ValueError("unknown UHF Hessian backend")
    return _optimize_step(key, X, hamiltonian, impls, batch_size=batch_size, cholesky=cholesky, lanczos_ncv=lanczos_ncv, lanczos_tol=lanczos_tol, static_spin_blocks=static_spin_blocks, uhf_backend=uhf_backend, hybrid_hessian_cached_batch_size=hybrid_hessian_cached_batch_size, hybrid_hessian_pair_batch_size=hybrid_hessian_pair_batch_size, hybrid_svd_rcond=hybrid_svd_rcond, hybrid_svd_max_degree=hybrid_svd_max_degree, hybrid_factorization=hybrid_factorization, nshards=jax.device_count())
