import numpy as np


def parity(x):
    """
    compute the parity of a permutation (along axis 1)
    vectorized along all leading batch dimensions

    this function is equivalent to
    ```
    from sympy.combinatorics.permutations import Permutation
    parity = partial(np.apply_along_axis, lambda x: Permutation(x).parity(), -1)
    ```
    """
    # see https://github.com/sympy/sympy/blob/96836db06ba6b78103dd4217db53db31502fb2f6/sympy/combinatorics/permutations.py#L115

    n = x.shape[-1]
    A = x[..., :, None] < x[..., None, :]
    mask = np.tril(np.ones((n, n), dtype=bool))
    mask = np.expand_dims(mask, tuple(range(0, A.ndim - 2)))
    return np.bitwise_xor.reduce(A & mask, axis=(-2, -1))
