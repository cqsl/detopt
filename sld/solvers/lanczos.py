# Derived from CuPy
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
# Licensed under the MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# code ported to jax from https://github.com/cupy/cupy/blob/03b45e5ce22599ce1f167a37124592660adf2620/cupyx/scipy/sparse/linalg/_eigen.py

import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import Partial


def nrm2inv(u):
    return jax.lax.rsqrt((u.conj() * u).sum().real)


def nrm2(u):
    return 1.0 / nrm2inv(u)


@partial(jax.jit, static_argnames=("k", "which", "ncv", "maxiter", "return_eigenvectors", "shape", "dtype"))
def eigsh(a, k=6, *, which="LM", v0=None, ncv=None, maxiter=None, tol=None, return_eigenvectors=True, key=None, shape=None, dtype=None):
    if isinstance(a, jax.Array):
        shape = a.shape
        dtype = a.dtype
        a = Partial(jnp.matmul, a)
    else:
        if shape is None or dtype is None:
            raise ValueError("need to pass shape and dtype")

    ndim = len(shape)

    n = shape[0]
    if ndim != 2 or shape[0] != shape[1]:
        raise ValueError(f"expected square matrix (shape: {shape})")
    if dtype.char not in "fdFD":
        raise TypeError(f"unsupprted dtype (actual: {dtype})")
    if k <= 0:
        raise ValueError(f"k must be greater than 0 (actual: {k})")
    if k >= n:
        raise ValueError(f"k must be smaller than n (actual: {k})")
    if which not in ("LM", "LA", "SA"):
        raise ValueError(f"which must be 'LM','LA'or'SA' (actual: {which})")
    if ncv is None:
        ncv = min(max(2 * k, k + 32), n - 1)
    else:
        ncv = min(max(ncv, k + 2), n - 1)
    if maxiter is None:
        maxiter = 10 * n
    if tol is None:
        tol = jnp.finfo(dtype).eps

    alpha = jnp.zeros((ncv,), dtype=dtype)
    beta = jnp.zeros((ncv,), dtype=dtype.char.lower())
    V = jnp.empty((ncv, n), dtype=dtype)

    # Set initial vector
    if v0 is None:
        if key is None:
            raise ValueError("need to pass key or v0")
        u = jax.random.uniform(key, (n,)).astype(dtype)
        V = V.at[0].set(u * nrm2inv(u))
    else:
        u = v0
        V = V.at[0].set(v0 * nrm2inv(v0))

    V, u, alpha, beta = _lanczos_fast(a, V, u, alpha, beta, 0, ncv)

    iter_ = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = V.T @ s

    # Compute residual
    beta_k = beta[-1] * s[-1, :]
    res = nrm2(beta_k)

    def cond_fun(val):
        V, u, alpha, beta, beta_k, x, res, iter_, w = val
        return jnp.logical_and(res > tol, iter_ < maxiter)

    def body_fun(val):
        V, u, alpha, beta, beta_k, x, res, iter_, w = val

        # Setup for thick-restart
        beta = beta.at[:k].set(0)
        alpha = alpha.at[:k].set(w)
        V = V.at[:k].set(x.T)

        u = u - u.T @ V[:k].conj().T @ V[:k]
        V = V.at[k].set(u * nrm2inv(u))

        u = a(V[k])
        alpha = alpha.at[k].set(V[k].conj() @ u)
        u = u - alpha[k] * V[k]
        u = u - V[:k].T.conj() @ beta_k
        tmp = nrm2inv(u)
        V = V.at[k + 1].set(u * tmp)
        beta = beta.at[k].set(1.0 / tmp)

        V, u, alpha, beta = _lanczos_fast(a, V, u, alpha, beta, k + 1, ncv)

        iter_ = iter_ + ncv - k
        w, s = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
        x = V.T @ s

        # Compute residual
        beta_k = beta[-1] * s[-1, :]
        res = nrm2(beta_k)

        return V, u, alpha, beta, beta_k, x, res, iter_, w

    *_, x, res, iter, w = jax.lax.while_loop(cond_fun, body_fun, (V, u, alpha, beta, beta_k, x, res, iter_, w))

    # TODO return info with res and iter

    if return_eigenvectors:
        idx = jnp.argsort(w)
        return w[idx], x[:, idx], (res, iter)
    else:
        return jnp.sort(w), (res, iter)


@jax.jit
def _lanczos_fast(a, V, u, alpha, beta, i_start, i_end):
    def body_fun(i, val):
        V, u, alpha, beta = val
        u = a(V[i])
        alpha = alpha.at[i].set(V[i].conj().T @ u)
        u = u - alpha[i] * V[i] - beta[i - 1] * V[i - 1]
        mask = jnp.arange(V.shape[0]) <= i
        u = u - jnp.einsum("j,kj,k,kl->l", u, V.conj(), mask, V)
        tmp = nrm2inv(u)
        beta = beta.at[i].set(1.0 / tmp)
        V = V.at[i + 1].set(jax.lax.select(i >= i_end - 1, V.at[i + 1].get(), u * tmp))
        return V, u, alpha, beta

    return jax.lax.fori_loop(i_start, i_end, body_fun, (V, u, alpha, beta))


@partial(jax.jit, static_argnums=(3, 4))
def _eigsh_solve_ritz(alpha, beta, beta_k, k, which):
    t = jnp.diag(alpha)
    t = t + jnp.diag(beta[:-1], k=1)
    t = t + jnp.diag(beta[:-1], k=-1)
    if beta_k is not None:
        t = t.at[k, :k].set(beta_k)
        t = t.at[:k, k].set(beta_k)
    w, s = jnp.linalg.eigh(t)

    if which == "LA":
        idx = jnp.argsort(w)
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == "LM":
        idx = jnp.argsort(jnp.abs(w))
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == "SA":
        idx = jnp.argsort(w)
        wk = w[idx[:k]]
        sk = s[:, idx[:k]]
    return wk, sk
