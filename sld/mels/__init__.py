from functools import partial
from ._impl import Impl
from jax.tree_util import Partial

from ._src.pairwise.derivatives import hess
from ._src.pairwise.derivatives_uhf import hess_spin, hess_spin_block, grad_spin

from ._src.pairwise import stable as stable_
from ._src.pairwise import unstable as unstable_

from ._src.integrate import integrate

from ._src.pairwise import compute_grad_weighted, compute_in_blocks, compute_mels, compute_mels_static_uhf_spin_blocks, static_uhf_spin_pattern
def get_impls(ghf, unstable):
    impl_ = unstable_ if unstable else stable_

    if ghf:
        hess_impl = partial(hess, impl_)
        hess_spin_blocks = None
        grad_impl = partial(integrate, hess_impl)
        expval_impl = impl_.mels.expval
        expval_sq_impl = impl_.mels.expval_sq
    else:
        hess_impl = partial(hess_spin, impl_)
        hess_spin_blocks = tuple(Partial(partial(hess_spin_block, impl_, spin_block=spin_block)) for spin_block in ("uu", "ud", "du", "dd"))
        grad_impl = partial(grad_spin, impl_)
        expval_impl = impl_.mels_uhf.expval_spin
        expval_sq_impl = impl_.mels_uhf.expval_sq_spin
    hess_impl = Partial(hess_impl)
    grad_impl = Partial(grad_impl)
    expval_impl = Partial(expval_impl)
    expval_sq_impl = Partial(expval_sq_impl)

    return Impl(hess=hess_impl, grad=grad_impl, expval=expval_impl, expval_sq=expval_sq_impl, hess_spin_blocks=hess_spin_blocks)


stable = get_impls(True, False)
unstable = get_impls(True, True)
stable_uhf = get_impls(False, False)
unstable_uhf = get_impls(False, True)

