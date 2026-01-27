import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def unpack_du(x):
    x_ = x.reshape(x.shape[:-1] + (2, -1))
    return x_[..., 0, :], x_[..., 1, :]


@partial(jnp.vectorize, signature="(n)->()", excluded=(0, 2, 3))
def slater_ansatz_index(Xi, n, transposed, slogdet_impl):
    # select the rows of the occupied orbitals
    if transposed:
        N_electrons, N_orbitals = Xi.shape
        (i,) = jnp.where(n == 1, size=N_electrons)
        A = Xi[:, i]
    else:
        N_orbitals, N_electrons = Xi.shape
        (i,) = jnp.where(n == 1, size=N_electrons)
        A = Xi[i]
    sgn, logdet = slogdet_impl(A)
    return logdet + jnp.log(sgn + 0.0j)


vslater_ansatz_index = jax.vmap(partial(slater_ansatz_index, slogdet_impl=jnp.linalg.slogdet, transposed=True), in_axes=(0, None))


def slater_ansatz_index_split_spin(Xi, Xi2, n, transposed, slogdet_impl):
    n_down, n_up = unpack_du(n)
    return slater_ansatz_index(Xi, n_down, transposed, slogdet_impl) + slater_ansatz_index(Xi2, n_up, transposed, slogdet_impl)


vslater_ansatz_index_split_spin = jax.vmap(partial(slater_ansatz_index_split_spin, slogdet_impl=jnp.linalg.slogdet, transposed=True), in_axes=(0, 0, None))
