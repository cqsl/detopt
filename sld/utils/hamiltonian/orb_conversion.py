import jax
import jax.numpy as jnp

import numpy as np
import sparse
from functools import partial

from sld.utils.parity import parity as _parity

interleave_default = False


def spinorb_from_spatial_sparse_coo2(hij_sparse, interleave=interleave_default, spin_values=[0, 1]):
    # Σ_ijσ h_ij c†_iσ c†_jσ
    # for σ ∈ spin_values
    # interleave=True -> 2i+spin
    # interleave=False -> n*spin+i

    if isinstance(spin_values, int):
        spin_values = list(spin_values)
    # assert spin_values in [[0], [1], [0,1]]

    assert isinstance(hij_sparse, sparse.COO)
    n = hij_sparse.shape[0]

    if interleave:
        t = lambda x: 2 * x
        tp1 = lambda x: 2 * x + 1
    else:
        t = lambda x: x
        tp1 = lambda x: x + n

    i, j = hij_sparse.coords
    a = hij_sparse.data
    a2 = []
    i2 = []
    j2 = []
    if 0 in spin_values:
        a2.append(a)
        i2.append(t(i))
        j2.append(t(j))
    if 1 in spin_values:
        a2.append(a)
        i2.append(tp1(i))
        j2.append(tp1(j))
    i2 = np.concatenate(i2)
    j2 = np.concatenate(j2)
    a2 = np.concatenate(a2)
    return sparse.COO(np.array([i2, j2]), a2, shape=(2 * n, 2 * n))


def spinorb_from_spatial_sparse_coo4(hijkl_sparse, interleave=interleave_default, _order_preserving=False):
    # Σ_ijklμσ h_ijkl c†_iμ c†_jσ c_kμ c_lσ (two spins can be kept the same)
    # interleave=True -> 2i+spin
    # interleave=False -> n*spin+i
    assert isinstance(hijkl_sparse, sparse.COO)
    n = hijkl_sparse.shape[0]

    if interleave:
        t = lambda x: 2 * x
        tp1 = lambda x: 2 * x + 1
    else:
        t = lambda x: x
        tp1 = lambda x: x + n

    p, q, r, s = hijkl_sparse.coords
    b = hijkl_sparse.data

    if _order_preserving:
        # swap p/q and r/s
        a2 = np.concatenate([-b, -b, b, b])
        p2 = np.concatenate([tp1(q), tp1(p), t(p), tp1(p)])
        q2 = np.concatenate([t(p), t(q), t(q), tp1(q)])
        r2 = np.concatenate([tp1(r), tp1(s), t(r), tp1(r)])
        s2 = np.concatenate([t(s), t(r), t(s), tp1(s)])
    else:
        # same as openfermion
        a2 = np.concatenate([b, b, b, b])
        p2 = np.concatenate([t(p), tp1(p), t(p), tp1(p)])
        q2 = np.concatenate([tp1(q), t(q), t(q), tp1(q)])
        r2 = np.concatenate([tp1(r), t(r), t(r), tp1(r)])
        s2 = np.concatenate([t(s), tp1(s), t(s), tp1(s)])

    return sparse.COO(np.array([p2, q2, r2, s2]), a2, shape=(2 * n, 2 * n, 2 * n, 2 * n))


def spinorb_from_spatial2(hij, hijkl, interleave=interleave_default):
    h2 = spinorb_from_spatial_sparse_coo2(sparse.COO.from_numpy(hij), interleave)
    h4 = spinorb_from_spatial_sparse_coo4(sparse.COO.from_numpy(hijkl), interleave)
    return h2.todense(), h4.todense()


def to_desc_order_sparse(vijkl_sparse, cutoff, set_zero_same=True):
    n = vijkl_sparse.ndim
    assert n % 2 == 0
    if n > 2:
        ij = vijkl_sparse.coords[: n // 2]
        kl = vijkl_sparse.coords[n // 2 :]
        a = vijkl_sparse.data.copy()

        perm_ij = np.argsort(-ij, axis=0)
        perm_kl = np.argsort(-kl, axis=0)

        ij_desc = ij[perm_ij, np.arange(ij.shape[1])]
        kl_desc = kl[perm_kl, np.arange(kl.shape[1])]

        a *= 1 - 2 * (_parity(perm_ij.T) ^ _parity(perm_kl.T))
        if set_zero_same:
            # set to zero / remove all those where we try to create / destroy two on the same orbital
            mask = (np.diff(ij_desc, axis=0) == 0).any(axis=0) | (np.diff(kl_desc, axis=0) == 0).any(axis=0)
            a = a[~mask]
            ij_desc = ij_desc[:, ~mask]
            kl_desc = kl_desc[:, ~mask]
        new_coords = np.array([*ij_desc, *kl_desc])
        # use coo to merge same indices
        vijkl_sparse = sparse.COO(new_coords, a, shape=vijkl_sparse.shape)
        # we might have some new almost zeros from the cancellations, make sure they are 0
        new_coords2 = vijkl_sparse.coords
        a2 = vijkl_sparse.data
        mask = np.abs(a2) > cutoff
        return sparse.COO(new_coords2[:, mask], a2[mask], shape=vijkl_sparse.shape)
    else:
        return vijkl_sparse


def to_desc_order(hijkl, cutoff):
    return to_desc_order_sparse(sparse.COO.from_numpy(hijkl), cutoff).todense()


def uhf2ghf(Xu, Xd):
    return jnp.block([[Xu, jnp.zeros_like(Xu)], [jnp.zeros_like(Xd), Xd]])


@partial(jax.jit, static_argnames="set_zero_same")
def to_desc_order_dense(vijkl, cutoff, set_zero_same=True):
    if not set_zero_same:
        raise NotImplementedError
    L = jnp.tril(jnp.ones((vijkl.shape[0], vijkl.shape[0]), dtype=vijkl.dtype), -1)
    res = jnp.einsum("ijkl,ij,kl->ijkl", vijkl, L, L)
    res += jnp.einsum("jilk,ij,kl->ijkl", vijkl, L, L)
    res += -jnp.einsum("ijlk,ij,kl->ijkl", vijkl, L, L)
    res += -jnp.einsum("jikl,ij,kl->ijkl", vijkl, L, L)
    return res
