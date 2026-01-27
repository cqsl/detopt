import jax
import jax.numpy as jnp

from ..common import ol
from .mels import rdm1_red

from sld.utils.sparse import einsum_maybe_sparse


def ol_deriv(Aconj, B, v=None, return_mat=False):
    # ci cj^
    M = Aconj @ B.T
    r1 = rdm1_red(Aconj, B, v=v)
    no = Aconj.shape[1]
    tmp = jnp.eye(no) if v is None else v
    res = tmp * ol(Aconj, B) - jnp.linalg.det(M) * r1
    if return_mat:
        # use svd here and not eig, since batched jax.lax.eig runs on cpu per default on small matrices,
        # and even if you set implementation='CUSOLVER' it is very slow
        _, ev, _ = jax.lax.linalg.svd(M)
        return res, (jnp.abs(ev).min() / jnp.abs(ev).max())[None, None]
    else:
        return res


def E1e_deriv(Aconj, B, *args, mo_coeff=None):
    *v, H1e = args

    if H1e is None:
        return 0

    (v,) = v if len(v) > 0 else (None,)

    # H = c^ c
    # ci H cj^
    ne, no = Aconj.shape
    M = Aconj @ B.T

    if H1e.ndim == 1:  # diagonal version for fermi-hubbard
        raise NotImplementedError
    else:
        if mo_coeff is not None:
            # we need to the full mo H1e anyway
            H1e = jnp.einsum("AB,Aa,Bb->ab", H1e, mo_coeff, mo_coeff)
        r1 = rdm1_red(Aconj, B)
        tmp1 = jnp.einsum("ba,ab", r1, H1e)
        if v is None:
            x1 = H1e
            x2 = jnp.eye(no) * tmp1
            x3 = -jnp.einsum("ga,ab->gb", r1, H1e)
            x4 = -jnp.einsum("be,gb->ge", r1, H1e)
            x5 = jnp.einsum("be,ga,ab->ge", r1, r1, H1e) - r1 * tmp1
        else:
            r1v = rdm1_red(Aconj, B, v=v)
            x1 = H1e @ v
            x2 = v * tmp1
            x3 = -jnp.einsum("ga,ab,b...->g...", r1, H1e, v)
            x4 = -jnp.einsum("b...,gb->g...", r1v, H1e)
            x5 = jnp.einsum("b...,ga,ab->g...", r1v, r1, H1e) - r1v * tmp1
    return jnp.linalg.det(M) * (x1 + x2 + x3 + x4 + x5)


def E2e_deriv(Aconj, B, *args, mo_coeff=None, unroll=False):
    *v, H2e = args

    if H2e is None:
        return 0

    (v,) = v if len(v) > 0 else (None,)

    # H = c^ c^ c c
    # <A| ci H cj^ |B>
    # if v is not None:
    # \sum_j <A| ci H cj^|B> v_j

    ne, no = B.shape

    M = Aconj @ B.T
    I = jnp.eye(no)

    r1 = rdm1_red(Aconj, B)

    if v is not None:
        r1v = rdm1_red(Aconj, B, v=v)

    if mo_coeff is None:
        r1aoao = r1moao = r1aomo = r1
    else:
        # convert r1 to AO and do next contractions in AO
        r1aoao = jnp.einsum("ab,Aa,Bb->AB", r1, mo_coeff, mo_coeff)
        r1moao = jnp.einsum("ab,Bb->aB", r1, mo_coeff)
        r1aomo = jnp.einsum("ab,Aa->Ab", r1, mo_coeff)

    if not unroll:
        # the pyscf integrals are calculated in chemists notation (ij|kl)
        # it has 8 fold symmetry: i<->j, k<->l and ij<->kl
        #
        # H2e is: iklj   ( -> integrals in physicist notation)
        # Ht is: iklj - ikjl
        # Htt is: iklj - ikjl - kilj + kijl

        if isinstance(H2e, tuple):
            H2e, Ht, Htt = H2e
        else:
            Ht = H2e - H2e.transpose(0, 1, 3, 2)
            Htt = Ht - Ht.transpose(1, 0, 2, 3)

        if v is None:
            tmp0 = jnp.einsum("GB,EA,ABGE", r1aoao, r1aoao, Ht) * (I - r1)
            tmp1 = jnp.einsum("EA,ABGE->BG", r1aoao, Htt)
            tmp2 = jnp.einsum("Er,GA,ABGE->Br", r1aomo, r1aoao, Htt)
            tmp3 = jnp.einsum("uA,EB,ABGE->uG", r1moao, r1aoao, Htt)
            tmp4 = jnp.einsum("EA,Gr,uB,ABGE->ur", r1aoao, r1aomo, r1moao, Htt)
        else:
            tmp0 = jnp.einsum("GB,EA,ABGE", r1aoao, r1aoao, Ht) * (v - r1v)
            if mo_coeff is None:
                tmp1 = jnp.einsum("EA,ABGE,G...->B...", r1aoao, Htt, v)
                tmp3 = jnp.einsum("uA,EB,ABGE,G...->u...", r1moao, r1aoao, Htt, v)
            else:
                tmp1 = jnp.einsum("EA,ABGE,Bb,Gg,g...->b...", r1aoao, Htt, mo_coeff, mo_coeff, v)
                tmp3 = jnp.einsum("uA,EB,ABGE,Gg,g...->u...", r1moao, r1aoao, Htt, mo_coeff, v)

            tmp2 = jnp.einsum("Er,GA,ABGE,r...->B...", r1aomo, r1aoao, Htt, v)
            if mo_coeff is not None:
                tmp2 = jnp.einsum("B...,Bb->b...", tmp2, mo_coeff)
            tmp4 = jnp.einsum("EA,Gr,uB,ABGE,r...->u...", r1aoao, r1aomo, r1moao, Htt, v)
    else:
        assert not isinstance(H2e, tuple)
        if v is not None:
            tmp0a = einsum_maybe_sparse("ABGE,GB,EA", H2e, r1aoao, r1aoao, notation="physicist")
            tmp0b = einsum_maybe_sparse("ABGE,EB,GA", H2e, r1aoao, r1aoao, notation="physicist")
            tmp0 = (tmp0a - tmp0b) * (I - r1)

            if mo_coeff is None:
                # TODO support ... and partial contraction in einsum_maybe_sparse
                # for now we hardcode the single dummy dim of v with x
                assert v.ndim == 2 and v.shape[-1] == 1  # not implemented otherwise
                tmp1a = einsum_maybe_sparse("ABGE,EA,G->B", H2e, r1aoao, v.ravel(), notation="physicist")[..., None]
                tmp1b = einsum_maybe_sparse("ABGE,GA,E->B", H2e, r1aoao, v.ravel(), notation="physicist")[..., None]
                tmp1c = einsum_maybe_sparse("ABGE,EB,G->A", H2e, r1aoao, v.ravel(), notation="physicist")[..., None]
                tmp1d = einsum_maybe_sparse("ABGE,GB,E->A", H2e, r1aoao, v.ravel(), notation="physicist")[..., None]
                tmp1 = tmp1a - tmp1b - tmp1c + tmp1d

                tmp3a = jnp.einsum("BG,uB,G...->u...", tmp1a, r1moao, v)
                tmp3b = jnp.einsum("BE,uB,E...->u...", tmp1b, r1moao, v)
                tmp3c = jnp.einsum("AG,uA,G...->u...", tmp1c, r1moao, v)
                tmp3d = jnp.einsum("AE,uA,E...->u...", tmp1d, r1moao, v)
                tmp3 = -tmp3a + tmp3b + tmp3c - tmp3d
            else:
                v_ao = jnp.einsum("Aa,a...->A...", mo_coeff, v)
                # TODO support ... and partial contraction in einsum_maybe_sparse
                # for now we hardcode the single dummy dim of v with x
                assert v_ao.ndim == 2 and v_ao.shape[-1] == 1  # not implemented otherwise
                tmp1a = einsum_maybe_sparse("ABGE,EA,G->B", H2e, r1aoao, v_ao.ravel(), notation="physicist")[..., None]
                tmp1b = einsum_maybe_sparse("ABGE,GA,E->B", H2e, r1aoao, v_ao.ravel(), notation="physicist")[..., None]
                tmp1c = einsum_maybe_sparse("ABGE,EB,G->A", H2e, r1aoao, v_ao.ravel(), notation="physicist")[..., None]
                tmp1d = einsum_maybe_sparse("ABGE,GB,E->A", H2e, r1aoao, v_ao.ravel(), notation="physicist")[..., None]
                tmp1 = tmp1a - tmp1b - tmp1c + tmp1d
                tmp1 = jnp.einsum("BG,Bb,Gg->bg", tmp1, mo_coeff, mo_coeff)

                tmp3a = jnp.einsum("BG,uB,Gg,g...->u...", tmp1a, r1moao, mo_coeff, v)
                tmp3b = jnp.einsum("BE,uB,Ee,e...->u...", tmp1b, r1moao, mo_coeff, v)
                tmp3c = jnp.einsum("AG,uA,Gg,g...->u...", tmp1c, r1moao, mo_coeff, v)
                tmp3d = jnp.einsum("AE,uA,Ee,e...->u...", tmp1d, r1moao, mo_coeff, v)
                tmp3 = -tmp3a + tmp3b + tmp3c - tmp3d

            tmp2a = jnp.einsum("BG,Gr,r...->B...", tmp1a, r1aomo, v)
            tmp2b = jnp.einsum("BE,Er,r...->B...", tmp1b, r1aomo, v)
            tmp2c = jnp.einsum("AG,Gr,r...->A...", tmp1c, r1aomo, v)
            tmp2d = jnp.einsum("AE,Er,r...->A...", tmp1d, r1aomo, v)
            tmp2 = -tmp2a + tmp2b + tmp2c - tmp2d
            if mo_coeff is not None:
                tmp2 = jnp.einsum("B...,Bb->b...", tmp2, mo_coeff)

            tmp4a = jnp.einsum("BG,Gr,uB,r...->u...", tmp1a, r1aomo, r1moao, v)
            tmp4b = jnp.einsum("BE,Er,uB,r...->u...", tmp1b, r1aomo, r1moao, v)
            tmp4c = jnp.einsum("AG,Gr,uA,r...->u...", tmp1c, r1aomo, r1moao, v)
            tmp4d = jnp.einsum("AE,Er,uA,r...->u...", tmp1d, r1aomo, r1moao, v)
            tmp4 = tmp4a - tmp4b - tmp4c + tmp4d
        else:
            # separate einsum, mainly intended for the sparse / decomp version
            tmp0a = einsum_maybe_sparse("ABGE,GB,EA", H2e, r1aoao, r1aoao, notation="physicist")
            tmp0b = einsum_maybe_sparse("ABGE,EB,GA", H2e, r1aoao, r1aoao, notation="physicist")
            tmp0 = (tmp0a - tmp0b) * (I - r1)

            tmp1a = einsum_maybe_sparse("ABGE,EA->BG", H2e, r1aoao, notation="physicist")
            tmp1b = einsum_maybe_sparse("ABGE,GA->BE", H2e, r1aoao, notation="physicist")
            tmp1c = einsum_maybe_sparse("ABGE,EB->AG", H2e, r1aoao, notation="physicist")
            tmp1d = einsum_maybe_sparse("ABGE,GB->AE", H2e, r1aoao, notation="physicist")
            tmp1 = tmp1a - tmp1b - tmp1c + tmp1d

            tmp2a = jnp.einsum("BG,Gr->Br", tmp1a, r1aomo)
            tmp2b = jnp.einsum("BE,Er->Br", tmp1b, r1aomo)
            tmp2c = jnp.einsum("AG,Gr->Ar", tmp1c, r1aomo)
            tmp2d = jnp.einsum("AE,Er->Ar", tmp1d, r1aomo)
            tmp2 = -tmp2a + tmp2b + tmp2c - tmp2d

            tmp3a = jnp.einsum("BG,uB->uG", tmp1a, r1moao)
            tmp3b = jnp.einsum("BE,uB->uE", tmp1b, r1moao)
            tmp3c = jnp.einsum("AG,uA->uG", tmp1c, r1moao)
            tmp3d = jnp.einsum("AE,uA->uE", tmp1d, r1moao)
            tmp3 = -tmp3a + tmp3b + tmp3c - tmp3d

            tmp4a = jnp.einsum("BG,Gr,uB->ur", tmp1a, r1aomo, r1moao)
            tmp4b = jnp.einsum("BE,Er,uB->ur", tmp1b, r1aomo, r1moao)
            tmp4c = jnp.einsum("AG,Gr,uA->ur", tmp1c, r1aomo, r1moao)
            tmp4d = jnp.einsum("AE,Er,uA->ur", tmp1d, r1aomo, r1moao)
            tmp4 = tmp4a - tmp4b - tmp4c + tmp4d

    if mo_coeff is None:
        pass
    else:
        if v is None:
            # convert result to MO
            tmp1 = jnp.einsum("BG,Bb,Gg->bg", tmp1, mo_coeff, mo_coeff)
            tmp2 = jnp.einsum("Br,Bb->br", tmp2, mo_coeff)
            tmp3 = jnp.einsum("uG,Gg->ug", tmp3, mo_coeff)
        else:
            # we already did it above
            pass

    D = tmp0 + tmp1 + tmp2 + tmp3 + tmp4

    return jnp.linalg.det(M) * D
