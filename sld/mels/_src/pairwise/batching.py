from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import reshard
from jax.tree_util import Partial
from tqdm.auto import tqdm


def _call(f, args, keywords, Xc, Y, arrays):
    if isinstance(Xc.nelec, tuple):
        return f(*args, Xc.Xu, Xc.Xd, Y.Xu, Y.Xd, (Xc.s, Y.s, Xc.i, Y.i), arrays, **keywords)
    return f(*args, Xc.X, Y.X, (Xc.i, Y.i), arrays, **keywords)


def _evaluate_pair(mel_impl, Xi, Yj, arrays):
    return _call(mel_impl.func, mel_impl.args, mel_impl.keywords, Xi.conj(), Yj, arrays)


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0, None))
def _evaluate_pairs(mel_impl, Xi, Yj, arrays):
    return _evaluate_pair(mel_impl, Xi, Yj, arrays)


@partial(jax.jit, static_argnames=("batch_size"))
def compute_mels(mel_impl, X, Y=None, arrays=None, batch_size=None):
    assert isinstance(mel_impl, Partial)
    f = mel_impl.func
    args = mel_impl.args
    keywords = mel_impl.keywords

    if arrays is not None and arrays.get("S", None) is not None:
        raise NotImplementedError("a nonorthogonal orbital metric is only implemented in the hybrid backends")

    if (batch_size is not None) and ("i" in jax.typeof(jax.tree.leaves(X)[0]).sharding.spec):
        raise NotImplementedError("sharded Xs are not yet supported when using batch_size, only Ys")

    if Y is None:
        Y = X

    if batch_size is not None:
        ket_batch_size = min(batch_size, len(Y))
        bra_batch_size = max(1, batch_size // ket_batch_size)

        @partial(jax.smap, axis_name="i", out_axes=2)
        def jvf(Xc, Y, arrays, args, keywords):
            def evaluate_bra(Xci):
                values = jax.lax.map(lambda Yj: _call(f, args, keywords, Xci, Yj, arrays), Y, batch_size=ket_batch_size)
                # (bra, bra-orbital, ket, ket-orbital) layout
                return jax.tree.map(lambda value: jnp.moveaxis(value, 0, 1), values)

            return jax.lax.map(evaluate_bra, Xc, batch_size=bra_batch_size)

    else:

        @partial(jax.vmap, in_axes=(0, None, None, None, None))
        @partial(jax.vmap, in_axes=(None, 0, None, None, None), out_axes=1)
        def jvf(Xc, Y, arrays, args, keywords):
            return _call(f, args, keywords, Xc, Y, arrays)

    return jvf(X.conj(), Y, arrays, args, keywords)


@partial(jax.jit, static_argnames=("batch_size",))
def compute_grad_weighted(grad_impl, X, Y, ket_weights, arrays=None, batch_size=None):
    # accumulate over batches of kets, contracting with the weights right away, so the (bra, orbital, ket) tensor is never stored
    ket_axis_sharded = "i" in jax.typeof(jax.tree.leaves(Y)[0]).sharding.spec
    if ket_axis_sharded != ("i" in jax.typeof(ket_weights).sharding.spec):
        raise ValueError("Y and ket_weights must have the same sharding")

    nstates = ket_weights.shape[1]
    norb_active = X.n_orbitals if not isinstance(X.nelec, tuple) else X.Xu.shape[-1]
    dtype = jnp.result_type(*jax.tree.leaves((X, Y, ket_weights, arrays)))

    @partial(jax.smap, axis_name="i", out_axes=(None, None))
    def evaluate_shard(X, Y, ket_weights, arrays):
        # Y and ket_weights are device-local here when sharded; reduce the local shard before communicating
        nbatch = min(len(Y), 256 if batch_size is None else batch_size)
        # pad to full batches with copies of the last ket and zero weights
        npad = -len(Y) % nbatch
        Y = jax.tree.map(lambda value: jnp.concatenate([value, jnp.repeat(value[-1:], npad, axis=0)]), Y)
        ket_weights = jnp.pad(ket_weights, ((0, npad), (0, 0)))
        nbatches = len(ket_weights) // nbatch
        Y_batches = jax.tree.map(lambda value: value.reshape(nbatches, nbatch, *value.shape[1:]), Y)
        weight_batches = ket_weights.reshape(nbatches, nbatch, nstates)

        def evaluate_bra(Xi):
            initial = (jnp.zeros((nstates, norb_active, 1), dtype=dtype), jnp.zeros((nstates, norb_active, 1), dtype=dtype))
            if ket_axis_sharded:
                initial = jax.tree.map(lambda value: jax.lax.pcast(value, ("i",), to="varying"), initial)

            def reduce_batch(current, inputs):
                Y_batch, weights_batch = inputs
                S_batch, H_batch, *_ = jax.vmap(lambda Yj: _evaluate_pair(grad_impl, Xi, Yj, arrays))(Y_batch)
                return (current[0] + jnp.einsum("qoi,qs->soi", S_batch, weights_batch), current[1] + jnp.einsum("qoi,qs->soi", H_batch, weights_batch)), None

            return jax.lax.scan(reduce_batch, initial, (Y_batches, weight_batches))[0]

        S, H = jax.lax.map(evaluate_bra, X, batch_size=1)
        # (bra, state, orbital, 1) -> (bra, orbital, state)
        reduced = (S.transpose(0, 2, 1, 3)[..., 0], H.transpose(0, 2, 1, 3)[..., 0])
        if ket_axis_sharded:
            return jax.lax.psum(reduced, "i")
        return reduced

    return evaluate_shard(X, Y, ket_weights, arrays)


def _static_uhf_layout(ndet, nshards):
    nshards = jax.device_count() if nshards is None else nshards
    if ndet % nshards != 0:
        raise ValueError("the number of UHF determinants must be divisible by the number of device shards")
    local_ndet = ndet // nshards
    if local_ndet < 2:
        raise ValueError("static UHF spin blocks require at least two determinants per device shard")
    return nshards, local_ndet, (local_ndet + 1) // 2


def static_uhf_spin_pattern(ndet, nshards=None):
    nshards, local_ndet, local_nup = _static_uhf_layout(ndet, nshards)
    local_pattern = np.concatenate([np.zeros(local_nup, dtype=np.int32), np.ones(local_ndet - local_nup, dtype=np.int32)])
    return jnp.asarray(np.tile(local_pattern, nshards))


def _evaluate_static_uhf_block(mel_impl, Xc, Y, arrays, batch_size, nket_global):
    pair = lambda Xi, Yj: _call(mel_impl.func, mel_impl.args, mel_impl.keywords, Xi, Yj, arrays)
    if batch_size is None:
        return jax.vmap(jax.vmap(pair, in_axes=(None, 0), out_axes=1), in_axes=(0, None))(Xc, Y)
    evaluate_bra = lambda Xi: jax.vmap(lambda Yj: pair(Xi, Yj), out_axes=1)(Y)
    return jax.lax.map(evaluate_bra, Xc, batch_size=max(1, batch_size // nket_global))


@partial(jax.jit, static_argnames=("batch_size", "nshards"))
def compute_mels_static_uhf_spin_blocks(hess_spin_blocks, X, arrays=None, batch_size=None, nshards=None):
    nshards, local_ndet, local_nup = _static_uhf_layout(X.ndet, nshards)
    local_ndown = local_ndet - local_nup
    determinant_layout = np.arange(X.ndet, dtype=np.int32).reshape(nshards, local_ndet)
    bra_up_indices = jnp.asarray(determinant_layout[:, :local_nup].reshape(-1))
    bra_down_indices = jnp.asarray(determinant_layout[:, local_nup:].reshape(-1))
    nup = local_nup * nshards
    ndown = local_ndown * nshards
    Y = reshard(X, jax.P("i"))

    @partial(jax.smap, axis_name="i", out_axes=2)
    def evaluate_shard(Xc, Y, arrays, hess_spin_blocks):
        Xcu = Xc[bra_up_indices]
        Xcd = Xc[bra_down_indices]
        Yu = Y[:local_nup]
        Yd = Y[local_nup:]
        Suu, Huu = _evaluate_static_uhf_block(hess_spin_blocks[0], Xcu, Yu, arrays, batch_size, nup)
        Sud, Hud = _evaluate_static_uhf_block(hess_spin_blocks[1], Xcu, Yd, arrays, batch_size, ndown)
        Sdu, Hdu = _evaluate_static_uhf_block(hess_spin_blocks[2], Xcd, Yu, arrays, batch_size, nup)
        Sdd, Hdd = _evaluate_static_uhf_block(hess_spin_blocks[3], Xcd, Yd, arrays, batch_size, ndown)

        def assemble(uu, ud, du, dd):
            up_rows = jnp.concatenate([uu, ud], axis=2)
            down_rows = jnp.concatenate([du, dd], axis=2)
            result = jnp.empty((X.ndet, X.n_orbitals, local_ndet, X.n_orbitals), dtype=uu.dtype)
            result = result.at[bra_up_indices].set(up_rows)
            return result.at[bra_down_indices].set(down_rows)

        return assemble(Suu, Sud, Sdu, Sdd), assemble(Huu, Hud, Hdu, Hdd)

    return evaluate_shard(X.conj(), Y, arrays, hess_spin_blocks)


def compute_in_blocks(f, X, Y=None, n_results=None, batch_size=None, arrays=None, hermitian=False, desc=None):
    if batch_size is None:
        return compute_mels(f, X, Y, arrays=arrays)
    hermitian = hermitian and Y is None
    Y = X if Y is None else Y
    if hermitian:
        pairs = np.stack(np.triu_indices(X.ndet), axis=1)
    else:
        pairs = np.indices((X.ndet, Y.ndet)).reshape(2, -1).T
    batch_size = min(batch_size, len(pairs))
    mesh = jax.sharding.get_abstract_mesh()

    result = [None] * n_results
    for start in tqdm(range(0, len(pairs), batch_size), desc=desc):
        i, j = pairs[start : start + batch_size].T
        # pad the last batch so it does not compile a second shape
        Xi, Yj = X[np.pad(i, (0, batch_size - len(i)), mode="edge")], Y[np.pad(j, (0, batch_size - len(j)), mode="edge")]
        if "i" in mesh.axis_names:
            Xi, Yj = reshard(Xi, jax.P("i")), reshard(Yj, jax.P("i"))
        values = _evaluate_pairs(f, Xi, Yj, arrays)
        if not mesh.empty:
            values = reshard(values, jax.P())
        for k in range(n_results):
            value = np.asarray(values[k])[: len(i)]
            if value.shape[1:] == (1, 1):
                value = value[:, 0, 0]
            if result[k] is None:
                result[k] = np.zeros((X.ndet, Y.ndet) + value.shape[1:], dtype=X.dtype)
            result[k][i, j] = value
            if hermitian:
                result[k][j, i] = (value.transpose(0, 2, 1) if value.ndim == 3 else value).conj()
    return tuple(result)
