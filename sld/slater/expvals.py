import jax
from functools import partial
from sld.mels import compute_mels


@partial(jax.jit, static_argnames="batch_size")
def compute_expval(X, op, expval_impl, batch_size=None):
    S, H, *_ = compute_mels(expval_impl, X, arrays=op, batch_size=batch_size)
    S = S[:, 0, :, 0]
    H = H[:, 0, :, 0]
    return (X.coeffs.conj() @ H @ X.coeffs) / (X.coeffs.conj() @ S @ X.coeffs)


# TODO default impls
@partial(jax.jit, static_argnames=("batch_size"))
def compute_fidelity(X, Y, op=None, batch_size=None, expval_impl=None, expval_sq_impl=None):
    Sxx, _ = compute_mels(expval_impl, X, arrays=None, batch_size=batch_size)
    Sxx = Sxx[:, 0, :, 0]
    _, Hxy = compute_mels(expval_impl, X, Y, arrays=op, batch_size=batch_size)
    Hxy = Hxy[:, 0, :, 0]
    nom = (X.coeffs.conj() @ Hxy @ Y.coeffs) * (Y.coeffs.conj() @ Hxy.T.conj() @ X.coeffs)
    _, H2yy, _ = compute_mels(expval_sq_impl, Y, arrays=op, batch_size=batch_size)
    H2yy = H2yy[:, 0, :, 0]
    denom = (X.coeffs.conj() @ Sxx @ X.coeffs) * (Y.coeffs.conj() @ H2yy @ Y.coeffs)
    return nom / denom
