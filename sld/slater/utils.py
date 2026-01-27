import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
@partial(jnp.vectorize, signature="(i,j)->(i,j)")
def orthogonalize_orbs(A):
    assert A.ndim == 2
    ne, no = A.shape
    assert no >= ne
    if ne == 0:
        return A
    q, r = jnp.linalg.qr(A.T)
    if jnp.iscomplexobj(A):
        return q.T * (r.diagonal().prod()) ** (1.0 / ne)
    else:
        d = r.diagonal()
        return q.T * jnp.exp(jnp.log(jnp.abs(d)).sum() / ne) * jnp.sign(d)[:, None]


@partial(jnp.vectorize, signature="(i,j)->(i,j),()")
def normalize_orbs(U):
    assert U.ndim == 2
    ne = U.shape[0]
    if ne == 0:
        return U, 1.0
    orb_normsq = (U * U.conj()).sum(axis=-1, keepdims=True)
    norm = jnp.exp(0.5 * jnp.log(orb_normsq).sum())
    return U * jax.lax.rsqrt(orb_normsq), norm


@partial(jnp.vectorize, signature="(i,j)->(i,j),()")
def normalize_det(U):
    assert U.ndim == 2
    ne = U.shape[0]
    if ne == 0:
        return U, 1.0
    orb_normsq = (U * U.conj()).sum(axis=1, keepdims=True)
    U = U * jax.lax.rsqrt(orb_normsq)
    sgn, logdet = jnp.linalg.slogdet(U @ U.T.conj())
    return U * jnp.exp(-logdet / ne / 2), orb_normsq.prod() * sgn * jnp.exp(logdet)


def sele1(idx, Xs):
    # return Xs[idx]
    if idx is None:
        return Xs
    else:
        return jnp.where(idx[:, None] == 0, Xs[0], Xs[1])


def sele2(idx, Xsa, Xsb):
    # return [Xsa, Xsb][idx]
    if idx is None:
        return Xsa
    else:
        return jnp.where(idx[:, None, None] == 0, Xsa[0], Xsb[0]), jnp.where(idx[:, None, None] == 1, Xsa[1], Xsb[1])
