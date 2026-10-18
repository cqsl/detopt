from functools import partial


def get_projected_expval_impl(ghf, unstable, ghf_operator=False, **kwargs):
    # imported here so that the package (e.g. projection_setup, fft) loads without jax
    from jax.tree_util import Partial
    from sld.mels import Impl
    from sld.mels._src.pairwise import stable as stable_
    from sld.mels._src.pairwise import unstable as unstable_
    from .kernels import expval_proj_ghf, expval_proj_uhf, expval_proj_uhf_via_ghf

    impl_ = unstable_ if unstable else stable_

    if ghf:
        expval_impl = partial(expval_proj_ghf, impl_.mels.expval)
    elif ghf_operator:
        expval_impl = partial(expval_proj_uhf_via_ghf, impl_.mels.expval)
    else:
        expval_impl = partial(expval_proj_uhf, impl_.mels_uhf.expval_spin)

    expval_impl = Partial(expval_impl, **kwargs)
    return Impl(expval=expval_impl)


def get_projected_hess_impl(ghf, unstable, ghf_operator=False, **kwargs):
    # the Hessian of ⟨Ĥ P̂⟩ and ⟨P̂⟩ for sld-optimize, pairwise; kwargs: weights, trafos
    from jax.tree_util import Partial
    from sld.mels import Impl
    from sld.mels._src.pairwise import stable as stable_
    from sld.mels._src.pairwise import unstable as unstable_
    from .kernels import hess_proj_ghf, hess_proj_uhf, hess_proj_uhf_via_ghf

    impl_ = unstable_ if unstable else stable_
    hess_impl = hess_proj_ghf if ghf else hess_proj_uhf_via_ghf if ghf_operator else hess_proj_uhf
    return Impl(hess=Partial(partial(hess_impl, impl_), **kwargs))
