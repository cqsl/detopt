import jax
from jax import P
from jax.sharding import reshard

from ._src.pairwise import compute_mels


MEM_FRACTION = 0.65
FALLBACK_BATCH = 2048
SAFETY = 1.2


def _mem_limit(device=None):
    device = jax.devices()[0] if device is None else device
    try:
        stats = device.memory_stats()
    except (AttributeError, RuntimeError):
        return None
    if not stats:
        return None
    for key in ("bytes_limit", "bytes_reservable_limit"):
        value = stats.get(key)
        if value is not None and int(value) > 0:
            return int(value)
    return None


def _mem(compiled):
    # (fixed, tmp) bytes
    m = compiled.memory_analysis()
    fixed = max(0, int(m.argument_size_in_bytes) + int(m.output_size_in_bytes) - int(m.alias_size_in_bytes))
    return fixed, max(0, int(m.temp_size_in_bytes))


def check_full_hessian_output_memory(ndet, n_orbitals, dtype_itemsize, nshards):
    limit = _mem_limit()
    if limit is None:
        return
    nbytes = 2 * int(ndet) ** 2 * int(n_orbitals) ** 2 * int(dtype_itemsize) // int(nshards)
    if nbytes > limit:
        raise MemoryError(
            f"full H and S outputs require {nbytes / 2**30:.1f} GiB per device; "
            f"JAX memory limit is {limit / 2**30:.1f} GiB. "
            "Use sld-optimize-partial with a smaller --partialopt."
        )


def auto_batch_size_from_lowering(lower, n, *, label, min_batch=1, mem_fraction=MEM_FRACTION, fallback=FALLBACK_BATCH, unit="pair batch"):
    if not 0 < mem_fraction < 1:
        raise ValueError("mem_fraction must be between 0 and 1")
    if n <= 0:
        raise ValueError("cannot tune an empty matrix")
    bmin = max(1, min(int(min_batch), n))
    limit = _mem_limit()
    if limit is None:
        b = min(n, max(bmin, fallback))
        print(f"automatic {label} {unit}: {b} (memory limit unknown)")
        return b

    target = int(mem_fraction * limit / SAFETY)
    cache = {}

    def total(b):
        b = max(bmin, min(n, int(b)))
        if b not in cache:
            cache[b] = sum(_mem(lower(b).compile()))
        return cache[b]

    fixed, tmp = _mem(lower(bmin).compile())
    cache[bmin] = fixed + tmp
    if fixed + tmp > target:
        print(f"automatic {label} {unit}: {bmin} (minimum, exceeds memory target)")
        return bmin

    # guess from the temporaries of the smallest batch, rounded down to bmin * 2^k
    units = n // bmin if tmp <= 0 else max(1, min(n // bmin, (target - fixed) // tmp))
    b = min(n, bmin * (1 << (int(units).bit_length() - 1)))

    while b > bmin and total(b) > target:
        b = max(bmin, b // 2 // bmin * bmin)
    while b < n and total(min(n, 2 * b)) <= target:
        b = min(n, 2 * b)

    print(f"automatic {label} {unit}: {b} ({total(b) / 2**30:.2f} GiB)")
    return b


def auto_batch_size(mel_impl, X, Y, arrays, *, label, mem_fraction=MEM_FRACTION, fallback=FALLBACK_BATCH, unit="pair batch"):
    nbra, nket = len(X), len(Y)
    if nbra * nket == 0:
        raise ValueError("cannot tune an empty matrix")
    lower = lambda b: compute_mels.lower(mel_impl, X, reshard(Y, P("i")), arrays=arrays, batch_size=b)
    rows = max(1, min(nbra, fallback // nket))
    return auto_batch_size_from_lowering(lower, nbra * nket, label=label, min_batch=nket, mem_fraction=mem_fraction, fallback=rows * nket, unit=unit)


def resolve_batch_sizes(batch_size, batch_size_expval, batch_size_grad, batch_size_hess, *, impls, X_expval, Y_expval, X_grad, Y_grad, X_hess, Y_hess, arrays):
    sizes = []
    for b, impl, X, Y, label in [(batch_size_expval, impls.expval, X_expval, Y_expval, "expectation-value"), (batch_size_grad, impls.grad, X_grad, Y_grad, "gradient"), (batch_size_hess, impls.hess, X_hess, Y_hess, "Hessian")]:
        b = batch_size if b is None else b
        sizes.append(auto_batch_size(impl, X, Y, arrays, label=label) if b is None else b)
    return tuple(sizes)


def auto_cached_batch_size(lower, nbra, *, label, mem_fraction=MEM_FRACTION):
    return auto_batch_size_from_lowering(lower, nbra, label=label, mem_fraction=mem_fraction, fallback=16, unit="cached batch")


__all__ = ["auto_batch_size", "auto_batch_size_from_lowering", "auto_cached_batch_size", "check_full_hessian_output_memory", "resolve_batch_sizes"]
