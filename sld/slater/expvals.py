import jax
from functools import partial
from sld.mels import compute_mels


@partial(jax.jit, static_argnames="batch_size")
def compute_expval(X, op, expval_impl, batch_size=None):
    S, H, *_ = compute_mels(expval_impl, X, arrays=op, batch_size=batch_size)
    S = S[:, 0, :, 0]
    H = H[:, 0, :, 0]
    return (X.coeffs.conj() @ H @ X.coeffs) / (X.coeffs.conj() @ S @ X.coeffs)
