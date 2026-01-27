import jax
from jax import P
from jax.sharding import reshard


import jax.numpy as jnp
from functools import partial

from sld.solvers import generalized_eigh_cholesky_lanczos_tensor, generalized_eigh_lowrank_tensor

from sld.mels import compute_mels


@partial(jax.jit, static_argnames=("batch_size", "cholesky"))
def _optimize_step(
    key,
    X,
    hamiltonian,
    impls,
    batch_size=None,
    cholesky=True,
):
    # cholesky: use cholesky+lanczos to solve the eigenvalue problem if True; use dense solver if False

    k1, k2 = jax.random.split(key)
    # we perfrom a random rotation mixing all the orbitals, so it is sufficient to optimize the first one.
    X = X.choose_random_orbitals_to_optimize(k1, optimize_first_only=True)
    # orthogonalize the orbitals; in particular this normalizes them (which might already be sufficient to ensure numerical stability)
    X = X.orthogonalize_orbitals()
    # orthogonalization and random unitary rotation are an element of the SL group
    X = X.randomly_rotate_orbitals(k2)

    if True:
        S, H = compute_mels(impls.hess, X, reshard(X, P("i")), arrays=hamiltonian, batch_size=batch_size)

        if cholesky:
            x0 = X.get_orbitals_weighted()
            e, v, info = generalized_eigh_cholesky_lanczos_tensor(H, S, nelec=X.nelec, x0=x0)
        else:
            H = reshard(H, jax.P())
            S = reshard(S, jax.P())
            e, v = generalized_eigh_lowrank_tensor(H, S, n_zero_eigvals=int(X.n_zero_eigvals), rcond=1e-13)
            info = None

    e0 = e[0]
    v0 = v[0]

    coeffs = jnp.linalg.norm(v0, axis=1)
    v0 = v0 / coeffs[:, None]
    X = X.set_orbitals(v0)
    X = X.replace(coeffs=coeffs)
    X = X.clear_orbitals_to_optimize()

    return X, e0, info


def optimize_step(key, X, hamiltonian, impls, batch_size=None, cholesky=True):
    if isinstance(X.nelec, tuple) and X.nelec[0] != X.nelec[1]:
        # non-jitted, since the rank depends on the spin chosen per det
        return _optimize_step.__wrapped__(key, X, hamiltonian, impls, batch_size=batch_size, cholesky=cholesky)
    else:
        return _optimize_step(key, X, hamiltonian, impls, batch_size=batch_size, cholesky=cholesky)
