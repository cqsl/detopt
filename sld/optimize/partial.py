import jax

from jax import P
from jax.sharding import reshard


import jax.numpy as jnp
from functools import partial

from sld.solvers import generalized_eigh_partial_cholesky_lanczos_tensor
from sld.mels import compute_mels

from .common import integrate_and_assemble_all


@partial(jax.jit, static_argnames=("ndet_partial", "batch_size", "cholesky"), donate_argnames=("S_all", "H_all"))
def optimize_step_partial(
    key,
    X,
    hamiltonian,
    impls,
    mo_coeff=None,
    ndet_partial=None,
    p_prob=None,
    batch_size=None,
    S_all=None,
    H_all=None,
    cholesky=True,
):
    # optimize only a subset of determinant's orbitals, lvp for the rest
    # choose randomly of which dets to optimize orbs, and of which only the linear coeff
    # p_prob: (optional) different prob for different dets

    # here we work with determinants where each orbital is assumed to be normalized to norm 1
    # this is done so that we can more easily update S_all and H_all
    # Further we keep the hole state normalized via coeffs (this is ensured automatically by the eigenvalue solver returning a normazlied eigenvector)

    k1, k2, k3 = jax.random.split(key, 3)
    k_full = jnp.sort(jax.random.choice(k1, X.ndet, shape=(ndet_partial,), replace=False, p=p_prob))
    k_lvp = jnp.delete(jnp.arange(X.ndet), k_full, assume_unique_indices=True)

    Xf = X[k_full]
    Xl = X[k_lvp]

    if S_all is None:
        assert H_all is None
        # normalize all determinants to orbital norm 1
        Xl = Xl.normalize_orbs()
        Sll, Hll = compute_mels(impls.expval, Xl, reshard(Xl, P("i")), arrays=hamiltonian, batch_size=batch_size)
        Sll = reshard(Sll, P())
        Hll = reshard(Hll, P())
    else:
        Sll = jnp.take_along_axis(jnp.take_along_axis(S_all, k_lvp[None, :], 1), k_lvp[:, None], 0)[:, None, :, None]
        Hll = jnp.take_along_axis(jnp.take_along_axis(H_all, k_lvp[None, :], 1), k_lvp[:, None], 0)[:, None, :, None]

    # we perfrom a random rotation mixing all the orbitals, so it is sufficient to optimize the first one.
    Xf = Xf.choose_random_orbitals_to_optimize(k2, optimize_first_only=True)
    # orthogonalize the orbitals; in particular this normalizes them (which might already be sufficient to ensure numerical stability)
    Xf = Xf.orthogonalize_orbitals()
    # orthogonalization and random unitary rotation are an element of the SL group
    Xf = Xf.randomly_rotate_orbitals(k3)

    Sff, Hff, *_ = compute_mels(impls.hess, Xf, reshard(Xf, P("i")), arrays=hamiltonian, batch_size=batch_size)
    Sfl, Hfl, *_ = compute_mels(impls.grad, Xf, reshard(Xl, P("i")), arrays=hamiltonian, batch_size=batch_size)

    x0f = Xf.get_orbitals_weighted()
    x0l = Xl.coeffs
    if cholesky:
        e0, v0f, v0l, info = generalized_eigh_partial_cholesky_lanczos_tensor(Sff, Hff, Sfl, Hfl, Sll, Hll, nelec=Xf.nelec, x0f=x0f, x0l=x0l)
    else:
        raise NotImplementedError
    coeffs_l = v0l.ravel()
    coeffs_f = jnp.linalg.norm(v0f, axis=1)
    v0f = v0f / coeffs_f[:, None]

    Xl = Xl.replace(coeffs=coeffs_l)

    Xf = Xf.set_orbitals(v0f)
    Xf = Xf.replace(coeffs=coeffs_f)
    Xf = Xf.clear_orbitals_to_optimize()

    # concatenate
    X = Xf + Xl
    S_all, H_all = integrate_and_assemble_all(v0f, Sff, Hff, Sfl, Hfl, Sll, Hll)

    return X, e0, info, S_all, H_all
