import jax.numpy as jnp


def einsum_maybe_sparse(subscripts, *operands, **kwargs):
    notation = kwargs.pop("notation", "chemist")  # noqa F841
    # add here custom code for contracting with a compressed version of the eri tensor, e.g. cholesky-decomposed or in a sparse format
    # if isinstance(operands[0], CholeskyArray):
    #     return einsum_cholesky(subscripts, *operands, notation=notation, **kwargs)
    return jnp.einsum(subscripts, *operands, **kwargs)
