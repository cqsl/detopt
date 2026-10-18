from functools import partial

import jax
import jax.numpy as jnp

from sld.symmetry_projection.kernels import _grid_apply, _resolve_trafos
from sld.utils.hamiltonian.orb_conversion import uhf2ghf

from .common import _unpack, metric
from .ghf import mels_batched as ghf
from .uhf import mels_batched as uhf


def _projection_grid(weights, trafos, angles, spatial):
    trafos, angles, spatial = _resolve_trafos(trafos, angles, spatial)
    weights = jnp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    nspatial = 1 if spatial is None else spatial.shape[0]
    ngrid = trafos.shape[0] if trafos is not None else angles.shape[0] * nspatial
    if weights.shape[0] != ngrid:
        raise ValueError("weights and the projection grid must have the same length")
    if ngrid == 0:
        raise ValueError("the projection grid must not be empty")
    return weights, trafos, angles, spatial, ngrid


def _project_ghf_kets(Bs, trafos, angles, spatial):
    if trafos is not None:
        return jnp.einsum("bej,gjk->gbek", Bs, trafos)
    return _grid_apply(Bs, angles, spatial)


def _project_uhf_kets(Bs, trafos):
    return jnp.einsum("gij,bej->gbei", trafos, Bs)


def _reduce_grid(values, weights, nbra, nket, ngrid):
    return jax.tree.map(
        lambda value: jnp.einsum("agj,g->aj", value.reshape(nbra, ngrid, nket), weights),
        values,
    )


def expval_proj_ghf_pairs(
    Aconjs,
    Bs,
    H1e=None,
    H2e=None,
    Enuc=0,
    *,
    weights,
    trafos=None,
    angles=None,
    spatial=None,
    rcond=0,
    max_degree=2,
    cached_batch_size=16,
    pair_batch_size=128,
    factorization="svd",
    S=None,
    progress=None,
):
    weights, trafos, angles, spatial, ngrid = _projection_grid(weights, trafos, angles, spatial)
    nbra, nket = Aconjs.shape[0], Bs.shape[0]
    projected = _project_ghf_kets(Bs, trafos, angles, spatial)
    projected = projected.reshape(ngrid * nket, *projected.shape[2:])
    values = ghf.expval_pairs(
        Aconjs,
        projected,
        H1e,
        H2e,
        Enuc,
        rcond,
        max_degree,
        cached_batch_size,
        pair_batch_size,
        factorization,
        S,
        progress,
    )
    return _reduce_grid(values, weights, nbra, nket, ngrid)


@partial(
    jax.jit,
    static_argnames=("max_degree", "factorization", "cached_batch_size", "pair_batch_size", "progress"),
)
def expval_proj_ghf(
    X,
    Y,
    arrays=None,
    *,
    weights,
    trafos=None,
    angles=None,
    spatial=None,
    rcond=0,
    max_degree=2,
    factorization="svd",
    cached_batch_size=16,
    pair_batch_size=128,
    progress=None,
):
    H1e, H2e, w, Enuc = _unpack(arrays, "projected GHF expectation value")
    if w is not None:
        raise NotImplementedError("w is a UHF opposite-spin operator; GHF folds it into H2e")
    return expval_proj_ghf_pairs(
        X.conj().X,
        Y.X,
        H1e,
        H2e,
        Enuc,
        weights=weights,
        trafos=trafos,
        angles=angles,
        spatial=spatial,
        rcond=rcond,
        max_degree=max_degree,
        cached_batch_size=cached_batch_size,
        pair_batch_size=pair_batch_size,
        factorization=factorization,
        S=metric(arrays),
        progress=progress,
    )


def expval_proj_uhf_via_ghf_pairs(
    Aconju,
    Aconjd,
    Bu,
    Bd,
    H1e=None,
    H2e=None,
    Enuc=0,
    *,
    weights,
    trafos=None,
    angles=None,
    spatial=None,
    rcond=0,
    max_degree=2,
    cached_batch_size=16,
    pair_batch_size=128,
    factorization="svd",
    S=None,
    progress=None,
):
    Aconjs = jax.vmap(uhf2ghf)(Aconju, Aconjd)
    Bs = jax.vmap(uhf2ghf)(Bu, Bd)
    if S is not None and S.shape[-1] * 2 == Bs.shape[-1]:
        zero = jnp.zeros_like(S)
        S = jnp.block([[S, zero], [zero, S]])
    return expval_proj_ghf_pairs(
        Aconjs,
        Bs,
        H1e,
        H2e,
        Enuc,
        weights=weights,
        trafos=trafos,
        angles=angles,
        spatial=spatial,
        rcond=rcond,
        max_degree=max_degree,
        cached_batch_size=cached_batch_size,
        pair_batch_size=pair_batch_size,
        factorization=factorization,
        S=S,
        progress=progress,
    )


@partial(
    jax.jit,
    static_argnames=("max_degree", "factorization", "cached_batch_size", "pair_batch_size", "progress"),
)
def expval_proj_uhf_via_ghf(
    X,
    Y,
    arrays=None,
    *,
    weights,
    trafos=None,
    angles=None,
    spatial=None,
    rcond=0,
    max_degree=2,
    factorization="svd",
    cached_batch_size=16,
    pair_batch_size=128,
    progress=None,
):
    H1e, H2e, w, Enuc = _unpack(
        arrays, "projected UHF-via-GHF expectation value"
    )
    if w is not None:
        raise NotImplementedError("the UHF-via-GHF path requires a GHF spin-orbital Hamiltonian")
    Xc = X.conj()
    return expval_proj_uhf_via_ghf_pairs(
        Xc.Xu,
        Xc.Xd,
        Y.Xu,
        Y.Xd,
        H1e,
        H2e,
        Enuc,
        weights=weights,
        trafos=trafos,
        angles=angles,
        spatial=spatial,
        rcond=rcond,
        max_degree=max_degree,
        cached_batch_size=cached_batch_size,
        pair_batch_size=pair_batch_size,
        factorization=factorization,
        S=metric(arrays),
        progress=progress,
    )


def expval_proj_uhf_pairs(
    Aconju,
    Aconjd,
    Bu,
    Bd,
    H1e=None,
    H2e=None,
    w=None,
    Enuc=0,
    *,
    weights,
    trafos,
    rcond=0,
    max_degree=2,
    cached_batch_size=16,
    pair_batch_size=128,
    factorization="svd",
    S=None,
    progress=None,
):
    weights, trafos, _, _, ngrid = _projection_grid(weights, jnp.asarray(trafos), None, None)
    nbra, nket = Aconju.shape[0], Bu.shape[0]
    projected_u = _project_uhf_kets(Bu, trafos)
    projected_d = _project_uhf_kets(Bd, trafos)
    projected_u = projected_u.reshape(ngrid * nket, *projected_u.shape[2:])
    projected_d = projected_d.reshape(ngrid * nket, *projected_d.shape[2:])
    values = uhf.expval_pairs(
        Aconju,
        Aconjd,
        projected_u,
        projected_d,
        H1e,
        H2e,
        w,
        Enuc,
        rcond,
        max_degree,
        cached_batch_size,
        pair_batch_size,
        factorization,
        None,
        S,
        progress,
    )
    return _reduce_grid(values, weights, nbra, nket, ngrid)


@partial(
    jax.jit,
    static_argnames=("max_degree", "factorization", "cached_batch_size", "pair_batch_size", "progress"),
)
def expval_proj_uhf(
    X,
    Y,
    arrays=None,
    *,
    weights,
    trafos,
    rcond=0,
    max_degree=2,
    factorization="svd",
    cached_batch_size=16,
    pair_batch_size=128,
    progress=None,
):
    H1e, H2e, w, Enuc = _unpack(arrays, "projected UHF expectation value")
    Xc = X.conj()
    return expval_proj_uhf_pairs(
        Xc.Xu,
        Xc.Xd,
        Y.Xu,
        Y.Xd,
        H1e,
        H2e,
        w,
        Enuc,
        weights=weights,
        trafos=trafos,
        rcond=rcond,
        max_degree=max_degree,
        cached_batch_size=cached_batch_size,
        pair_batch_size=pair_batch_size,
        factorization=factorization,
        S=metric(arrays),
        progress=progress,
    )


def expval_sq_proj_uhf_pairs(
    Aconju,
    Aconjd,
    Bu,
    Bd,
    H1e=None,
    H2e=None,
    w=None,
    Enuc=0,
    *,
    weights,
    trafos,
    rcond=1e-10,
    max_degree=4,
    cached_batch_size=16,
    pair_batch_size=128,
    factorization="svd",
    S=None,
    progress=None,
):
    weights, trafos, _, _, ngrid = _projection_grid(weights, jnp.asarray(trafos), None, None)
    nbra, nket = Aconju.shape[0], Bu.shape[0]
    projected_u = _project_uhf_kets(Bu, trafos).reshape(ngrid * nket, *Bu.shape[1:])
    projected_d = _project_uhf_kets(Bd, trafos).reshape(ngrid * nket, *Bd.shape[1:])
    values = uhf.expval_sq_pairs(
        Aconju,
        Aconjd,
        projected_u,
        projected_d,
        H1e,
        H2e,
        w,
        Enuc,
        rcond,
        max_degree,
        cached_batch_size,
        pair_batch_size,
        factorization,
        S,
        progress,
    )
    return _reduce_grid(values, weights, nbra, nket, ngrid)


@partial(
    jax.jit,
    static_argnames=("max_degree", "factorization", "cached_batch_size", "pair_batch_size", "progress"),
)
def expval_sq_proj_uhf(
    X,
    Y,
    arrays=None,
    *,
    weights,
    trafos,
    rcond=1e-10,
    max_degree=4,
    factorization="svd",
    cached_batch_size=16,
    pair_batch_size=128,
    progress=None,
):
    H1e, H2e, w, Enuc = _unpack(arrays, "projected UHF squared expectation value")
    Xc = X.conj()
    return expval_sq_proj_uhf_pairs(
        Xc.Xu,
        Xc.Xd,
        Y.Xu,
        Y.Xd,
        H1e,
        H2e,
        w,
        Enuc,
        weights=weights,
        trafos=trafos,
        rcond=rcond,
        max_degree=max_degree,
        cached_batch_size=cached_batch_size,
        pair_batch_size=pair_batch_size,
        factorization=factorization,
        S=metric(arrays),
        progress=progress,
    )


def expval_sq_proj_ghf_pairs(
    Aconjs,
    Bs,
    H1e=None,
    H2e=None,
    Enuc=0,
    *,
    weights,
    trafos=None,
    angles=None,
    spatial=None,
    rcond=1e-10,
    max_degree=4,
    cached_batch_size=16,
    pair_batch_size=128,
    factorization="svd",
    S=None,
    progress=None,
):
    weights, trafos, angles, spatial, ngrid = _projection_grid(weights, trafos, angles, spatial)
    nbra, nket = Aconjs.shape[0], Bs.shape[0]
    projected = _project_ghf_kets(Bs, trafos, angles, spatial)
    projected = projected.reshape(ngrid * nket, *projected.shape[2:])
    values = ghf.expval_sq_pairs(
        Aconjs,
        projected,
        H1e,
        H2e,
        Enuc,
        rcond,
        max_degree,
        cached_batch_size,
        pair_batch_size,
        factorization,
        S,
        progress,
    )
    return _reduce_grid(values, weights, nbra, nket, ngrid)


def expval_sq_proj_uhf_via_ghf_pairs(
    Aconju,
    Aconjd,
    Bu,
    Bd,
    H1e=None,
    H2e=None,
    Enuc=0,
    *,
    weights,
    trafos=None,
    angles=None,
    spatial=None,
    rcond=1e-10,
    max_degree=4,
    cached_batch_size=16,
    pair_batch_size=128,
    factorization="svd",
    S=None,
    progress=None,
):
    Aconjs = jax.vmap(uhf2ghf)(Aconju, Aconjd)
    Bs = jax.vmap(uhf2ghf)(Bu, Bd)
    if S is not None and S.shape[-1] * 2 == Bs.shape[-1]:
        zero = jnp.zeros_like(S)
        S = jnp.block([[S, zero], [zero, S]])
    return expval_sq_proj_ghf_pairs(
        Aconjs,
        Bs,
        H1e,
        H2e,
        Enuc,
        weights=weights,
        trafos=trafos,
        angles=angles,
        spatial=spatial,
        rcond=rcond,
        max_degree=max_degree,
        cached_batch_size=cached_batch_size,
        pair_batch_size=pair_batch_size,
        factorization=factorization,
        S=S,
        progress=progress,
    )


@partial(
    jax.jit,
    static_argnames=("max_degree", "factorization", "cached_batch_size", "pair_batch_size", "progress"),
)
def expval_sq_proj_ghf(
    X, Y, arrays=None, *, weights, trafos=None, angles=None, spatial=None,
    rcond=1e-10, max_degree=4, factorization="svd", cached_batch_size=16, pair_batch_size=128, progress=None,
):
    H1e, H2e, w, Enuc = _unpack(arrays, "projected GHF squared expectation value")
    if w is not None:
        raise NotImplementedError("w is a UHF opposite-spin operator; GHF folds it into H2e")
    return expval_sq_proj_ghf_pairs(
        X.conj().X, Y.X, H1e, H2e, Enuc, weights=weights, trafos=trafos, angles=angles,
        spatial=spatial, rcond=rcond, max_degree=max_degree, cached_batch_size=cached_batch_size,
        pair_batch_size=pair_batch_size, factorization=factorization, S=metric(arrays), progress=progress,
    )


@partial(
    jax.jit,
    static_argnames=("max_degree", "factorization", "cached_batch_size", "pair_batch_size", "progress"),
)
def expval_sq_proj_uhf_via_ghf(
    X, Y, arrays=None, *, weights, trafos=None, angles=None, spatial=None,
    rcond=1e-10, max_degree=4, factorization="svd", cached_batch_size=16, pair_batch_size=128, progress=None,
):
    H1e, H2e, w, Enuc = _unpack(arrays, "projected UHF-via-GHF squared expectation value")
    if w is not None:
        raise NotImplementedError("w is a UHF opposite-spin operator; GHF folds it into H2e")
    Xc = X.conj()
    return expval_sq_proj_uhf_via_ghf_pairs(
        Xc.Xu, Xc.Xd, Y.Xu, Y.Xd, H1e, H2e, Enuc, weights=weights, trafos=trafos, angles=angles,
        spatial=spatial, rcond=rcond, max_degree=max_degree, cached_batch_size=cached_batch_size,
        pair_batch_size=pair_batch_size, factorization=factorization, S=metric(arrays), progress=progress,
    )
