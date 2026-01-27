from functools import partial
from ._impl import Impl
from jax.tree_util import Partial

from ._src.hand_optimized.derivatives_uhf import hess, hess_spin, grad_spin

from ._src.hand_optimized import stable as stable_
from ._src.hand_optimized import unstable as unstable_

from ._src.integrate import integrate

from ._src.helpers import compute_mels


def get_impls(ghf, unstable, sparse_, **kwargs):
    # kwargs: pass mo_coeff here for the AO operators

    impl_ = unstable_ if unstable else stable_

    if ghf:
        hess_impl = partial(hess, impl_)
        grad_impl = partial(integrate, hess_impl)
        expval_impl = impl_.mels.expval
        expval_sq_impl = impl_.mels.expval_sq
        if sparse_:
            raise NotImplementedError
    else:
        hess_impl = partial(hess_spin, impl_)
        grad_impl = partial(grad_spin, impl_)
        expval_impl = impl_.mels_uhf.expval_spin
        expval_sq_impl = impl_.mels_uhf.expval_sq_spin
        if sparse_:
            hess_impl = partial(hess_impl, unroll=True)
            grad_impl = partial(grad_impl, unroll=True)
            expval_impl = partial(expval_impl, unroll=True)
            expval_sq_impl = partial(expval_sq_impl, unroll=True)

    hess_impl = Partial(hess_impl, **kwargs)
    grad_impl = Partial(grad_impl, **kwargs)
    expval_impl = Partial(expval_impl, **kwargs)
    expval_sq_impl = Partial(expval_sq_impl, **kwargs)

    return Impl(hess_impl, grad_impl, expval_impl, expval_sq_impl)


stable = get_impls(True, False, False)
unstable = get_impls(True, True, False)
stable_uhf = get_impls(False, False, False)
unstable_uhf = get_impls(False, True, False)
