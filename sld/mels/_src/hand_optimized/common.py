import jax.numpy as jnp


def ol(Uconj, V):
    A = Uconj @ V.T
    return jnp.linalg.det(A)


def red(f, Aconj, k, B, l, *args, grad=False, **kwargs):
    if grad:
        # keep dummy trailing axis
        v = B.at[(l,), :].get().T
        args = v, *args
    Aconj = jnp.delete(Aconj, k, axis=0, assume_unique_indices=True)
    B = jnp.delete(B, l, axis=0, assume_unique_indices=True)
    res = f(Aconj, B, *args, **kwargs)
    assert jnp.issubdtype(k.dtype, jnp.signedinteger)
    sign = 1 - 2 * ((k + l) % 2)  # (-1)**(k+l)
    if isinstance(res, tuple):
        res, *rest = res
        return sign * res, *rest
    else:
        return sign * res
