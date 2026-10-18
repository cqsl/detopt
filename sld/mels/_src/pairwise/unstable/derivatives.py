import jax.numpy as jnp

from ..common import antisymmetrized
from .mels import inv_and_det, rdm1_red_from_inverse


def ol_deriv(Aconj, B, v=None):
    # ci cj^
    M = Aconj @ B.T
    M_inv, determinant = inv_and_det(M)
    r1 = rdm1_red_from_inverse(Aconj, B, M_inv, v=v)
    no = Aconj.shape[1]
    tmp = jnp.eye(no) if v is None else v
    return determinant * (tmp - r1)


def E1e_deriv(Aconj, B, *args):
    *v, H1e = args

    if H1e is None:
        return 0

    (v,) = v if len(v) > 0 else (None,)

    # H = c^ c
    # ci H cj^
    no = Aconj.shape[1]
    M = Aconj @ B.T
    M_inv, determinant = inv_and_det(M)

    if H1e.ndim == 1:
        raise NotImplementedError("diagonal H1e in the unstable derivatives")
    else:
        r1 = rdm1_red_from_inverse(Aconj, B, M_inv)
        if H1e.shape[0] != r1.shape[0]:
            raise NotImplementedError("the unstable derivatives need H1e in the full orbital space")
        energy = jnp.einsum("ba,ab", r1, H1e)
        if v is None:
            r1h = jnp.eye(no, dtype=r1.dtype) - r1
            D = r1h @ H1e @ r1h + energy * r1h
        else:
            r1hv = v - r1 @ v
            Hr1hv = H1e @ r1hv
            D = (Hr1hv - r1 @ Hr1hv) + energy * r1hv
    return determinant * D


def E2e_deriv(Aconj, B, *args):
    *v, H2e = args

    if H2e is None:
        return 0

    (v,) = v if len(v) > 0 else (None,)

    # H = c^ c^ c c
    # <A| ci H cj^ |B>
    # if v is not None:
    # \sum_j <A| ci H cj^|B> v_j

    no = B.shape[1]

    M = Aconj @ B.T
    M_inv, determinant = inv_and_det(M)
    I = jnp.eye(no)

    r1 = rdm1_red_from_inverse(Aconj, B, M_inv)

    _, _, Htt = antisymmetrized(H2e)

    # D = r1h H1eff r1h + energy * r1h, with r1h = 1 - r1 and H1eff_BG = \sum_EA r1_EA Htt_ABGE
    H1eff = jnp.einsum("EA,ABGE->BG", r1, Htt)
    energy = 0.5 * jnp.einsum("GB,BG", r1, H1eff)

    if v is None:
        r1h = I - r1
        D = r1h @ H1eff @ r1h + energy * r1h
    else:
        r1hv = v - r1 @ v
        Hr1hv = H1eff @ r1hv
        D = (Hr1hv - r1 @ Hr1hv) + energy * r1hv

    return determinant * D
