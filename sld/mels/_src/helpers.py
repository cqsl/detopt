import jax
from functools import partial
from jax.tree_util import Partial


@partial(jax.jit, static_argnames=("batch_size"))
def compute_mels(mel_impl, X, Y=None, arrays=None, batch_size=None):
    assert isinstance(mel_impl, Partial)
    f = mel_impl.func
    args = mel_impl.args
    keywords = mel_impl.keywords

    if (batch_size is not None) and ("i" in jax.typeof(jax.tree.leaves(X)[0]).sharding.spec):
        raise NotImplementedError("sharded Xs are not yet supported when using batch_size, only Ys")

    if Y is None:
        # TODO triangular impl
        Y = X

    if batch_size is not None:
        batch_size_effective = batch_size // len(X)
        if batch_size_effective == 0:
            batch_size_effective = None

        @partial(jax.smap, axis_name="i", out_axes=2)
        def jvf(Xc, Y, arrays, args, keywords):
            @partial(jax.vmap, in_axes=(0, None, None), out_axes=1)
            def _jvf(Y, arrays, Xc):
                if isinstance(Xc.nelec, tuple):
                    return f(*args, Xc.Xu, Xc.Xd, Y.Xu, Y.Xd, (Xc.s, Y.s, Xc.i, Y.i), arrays, **keywords)
                else:
                    return f(*args, Xc.X, Y.X, (Xc.i, Y.i), arrays, **keywords)

            return jax.lax.map(Partial(_jvf, Y, arrays), Xc, batch_size=batch_size_effective)

    else:

        @partial(jax.vmap, in_axes=(0, None, None, None, None))
        @partial(jax.vmap, in_axes=(None, 0, None, None, None), out_axes=1)
        def jvf(Xc, Y, arrays, args, keywords):
            if isinstance(Xc.nelec, tuple):
                return f(*args, Xc.Xu, Xc.Xd, Y.Xu, Y.Xd, (Xc.s, Y.s, Xc.i, Y.i), arrays, **keywords)
            else:
                return f(*args, Xc.X, Y.X, (Xc.i, Y.i), arrays, **keywords)

    return jvf(X.conj(), Y, arrays, args, keywords)
