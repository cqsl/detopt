import jax
import jax.numpy as jnp


def integrate(hess_impl, Uconj, V, deriv_indices, hamiltonian=None, twice=False, **kwargs):
    # helper function to use the hessian code for gradients and expectation values

    # we need to factor out an arbitrary orbital for the ones we integrate.
    # pick all zeros.
    k, l = deriv_indices
    if l is None:
        l = jnp.zeros((), dtype=jnp.int32)
    if twice and k is None:
        k = jnp.zeros((), dtype=jnp.int32)
    deriv_indices = k, l

    # hessians
    S, H = hess_impl(Uconj, V, deriv_indices, hamiltonian, **kwargs)
    v = V[l][:, None]
    if twice:  # energy and overlap
        w = Uconj[k][:, None]
        return w.T @ (S @ v), w.T @ (H @ v)
    else:  # energy and overlap gradient
        return S @ v, H @ v


def integrate_uhf(mathbb_S_H_spin_impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian=None, twice=False, **kwargs):
    # helper function to use the hessian code for gradients and expectation values

    # we need to factor out an arbitrary orbital for the ones we integrate.
    # pick all zeros.
    s, t, k, l = deriv_indices
    if l is None:
        l = jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.int32)
    if t is None:
        t = jnp.zeros((), dtype=jnp.int32)
    if twice and k is None:
        k = jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.int32)
    if twice and s is None:
        s = jnp.zeros((), dtype=jnp.int32)
    deriv_indices = s, t, k, l

    # hessians
    S, H, *aux = mathbb_S_H_spin_impl(Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, **kwargs)

    lu, ld = l
    v = jax.lax.select(t, Vd[ld], Vu[lu])[:, None]
    if twice:  # energy and overlap
        ku, kd = k
        # the lhs orbital has the spin s of the bra, the rhs one the spin t of the ket
        w = jax.lax.select(s, Uconjd[kd], Uconju[ku])[:, None]
        return w.T @ (S @ v), w.T @ (H @ v), *aux
    else:  # energy and overlap gradient
        return S @ v, H @ v, *aux
