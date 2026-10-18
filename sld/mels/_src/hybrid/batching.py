import jax
import jax.numpy as jnp
from jax import P


def tick(progress):
    if progress is not None:
        jax.debug.callback(progress, ordered=False)


def batch_map(function, xs, batch_size):
    if batch_size is None:
        return jax.vmap(function)(xs)
    if batch_size <= 0:
        raise ValueError('batch sizes must be positive')
    leading = xs[0].shape[0] if isinstance(xs, tuple) else xs.shape[0]
    return jax.lax.map(function, xs, batch_size=min(batch_size, leading))


def _mesh():
    mesh = jax.sharding.get_abstract_mesh()
    if 'i' in mesh.axis_names:
        return mesh, mesh.shape['i']
    if jax.device_count() > 1:
        raise ValueError("sharded hybrid backends require a mesh axis named 'i'")
    return mesh, 1


def _check_sharding(value, expected, what):
    def canonical(spec):
        axes = list(spec)
        while axes and axes[-1] is None:
            axes.pop()
        return tuple(axes)

    expected_canonical = canonical(expected)
    for leaf in jax.tree.leaves(value):
        sharding = getattr(jax.typeof(leaf), 'sharding', None)
        actual = getattr(sharding, 'spec', None)
        if actual is not None and canonical(actual) != expected_canonical:
            raise ValueError(f"{what} must have sharding {expected}, got {actual}")
    return value


def _check_replicated(*parts):
    return _check_sharding(parts, P(), 'Hamiltonian arrays')


def _sector_first(up, when_up, when_down=None):
    when_down = when_up if when_down is None else when_down
    return jnp.where(up, when_up[0], when_down[1]), jnp.where(up, when_up[1], when_down[0])
