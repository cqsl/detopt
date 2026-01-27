import numpy as np
from itertools import permutations
from .parity import parity as _parity


def Sn(abc):
    res = []
    n = len(abc)
    for σ in np.array(list(permutations(np.arange(n)))):
        s = "".join(np.array(list(abc))[σ])
        sgn = (-1) ** _parity(σ)
        res.append([sgn, s])
    return res
