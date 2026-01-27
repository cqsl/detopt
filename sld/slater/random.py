import jax
import jax.numpy as jnp
from functools import partial

from .utils import normalize_det


@partial(jax.jit, static_argnames=("dim", "shape", "dtype", "su"))
def random_unitary_matrix(key, dim, shape=(), dtype=complex, su=False):
    # Random size N-dimensional matrices, dimension  (*shape, dim, dim)
    z = jax.random.normal(key, shape=shape + (dim, dim), dtype=dtype)
    q, r = jnp.linalg.qr(z)
    d = r.diagonal(offset=0, axis1=-2, axis2=-1)
    q *= jnp.sign(d)[..., None, :]
    if su:
        q = q * jnp.linalg.det(q)[..., None, None] ** (-1.0 / dim)
    return q


def random_dets(key, ndet, nelec, n_orbitals, dtype=complex, coeff=200, stddev=0.001, normalize=True):
    key, k0, k1, k2, k3 = jax.random.split(key, 5)

    real_dtype = jnp.zeros((0,), dtype=dtype).real.dtype
    cplx = jnp.issubdtype(dtype, jnp.complexfloating)
    Xs = coeff * jax.nn.initializers.normal(stddev)(k0, shape=(ndet, nelec, n_orbitals), dtype=real_dtype)
    if cplx:
        Xs = Xs + 1.0j * coeff * jax.nn.initializers.normal(stddev)(k2, shape=Xs.shape, dtype=real_dtype)
    if normalize:
        Xs, _ = normalize_det(Xs)
    return Xs


@jax.jit
def randomly_rotate_orbitals(key, Xs):
    k1, k2 = jax.random.split(key)
    ghf = not isinstance(Xs, tuple)
    if not ghf:  # UHF
        U0 = random_unitary_matrix(k1, Xs[0].shape[-2], Xs[0].shape[:-2], dtype=Xs[0].dtype)
        U1 = random_unitary_matrix(k2, Xs[1].shape[-2], Xs[1].shape[:-2], dtype=Xs[1].dtype)
        Xs = jnp.einsum("...ij,...ik->...jk", U0, Xs[0]), jnp.einsum("...ij,...ik->...jk", U1, Xs[1])
    else:
        U = random_unitary_matrix(k1, Xs.shape[-2], Xs.shape[:-2], dtype=Xs.dtype)
        Xs = jnp.einsum("...ij,...ik->...jk", U, Xs)
    return Xs
