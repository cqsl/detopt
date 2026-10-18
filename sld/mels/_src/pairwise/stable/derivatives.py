import jax.numpy as jnp

from .common import T2
from ..common import antisymmetrized
from sld.utils.antisymm import Sn


def ol_deriv(Aconj, B, v=None):
    # d Aconj dB overlap
    no = B.shape[1]

    M = Aconj @ B.T
    I = jnp.eye(no)
    X, Σ, YH = jnp.linalg.svd(M)
    D = jnp.linalg.det(X) * jnp.linalg.det(YH)
    Xc, YHc = X.conj(), YH.conj()
    t1 = T2(Σ, 1)

    if v is None:
        z = Σ.prod() * I - jnp.einsum("c,jc,jk,ci,ig->gk", t1, Xc, Aconj, YHc, B, optimize="optimal")
    else:
        z = Σ.prod() * v - jnp.einsum("c,jc,jk,k...,ci,ig->g...", t1, Xc, Aconj, v, YHc, B, optimize="optimal")
    return D * z


def E1e_deriv(Aconj, B, *args):
    *v, H1e = args
    (v,) = v if len(v) > 0 else (None,)

    if H1e is None:
        return 0

    # d Aconj dB E1
    # where E1 is c^\dagger c (normal ordering)
    no = B.shape[1]

    M = Aconj @ B.T
    I = jnp.eye(no)

    X, Σ, YH = jnp.linalg.svd(M)
    D = jnp.linalg.det(X) * jnp.linalg.det(YH)
    Xc, YHc = X.conj(), YH.conj()
    t1 = T2(Σ, 1)
    t2 = T2(Σ, 2)

    AH = Aconj @ H1e                            # "A H1e",  (ne, no)
    HB = B @ H1e.T                              # "H1e B",  (ne, no)
    AHB = AH @ B.T                              # (ne, ne), H1e reduced to ne

    adj = jnp.einsum("ci,c,jc->ij", YHc, t1, Xc, optimize="optimal")

    G = jnp.einsum("jo,ji,pi->op", Xc, AHB, YHc, optimize="optimal")
    core5 = jnp.einsum("oi,o,jo->ij", YHc, t2 @ jnp.diagonal(G), Xc, optimize="optimal") \
           -jnp.einsum("oi,op,jp->ij", YHc, t2 * G, Xc, optimize="optimal")
    z2_scalar = -jnp.einsum("ij,ji->", adj, AHB, optimize="optimal")

    if v is not None:
        AHv = jnp.einsum("je,e...->j...", AH, v, optimize="optimal")
        Av = jnp.einsum("je,e...->j...", Aconj, v, optimize="optimal")
        z1 = -Σ.prod() * H1e @ v
        z2 = z2_scalar * v
        z3 = jnp.einsum("ig,ij,j...->g...", B, adj, AHv, optimize="optimal")
        z4 = jnp.einsum("ig,ij,j...->g...", HB, adj, Av, optimize="optimal")
        z5 = jnp.einsum("ig,ij,j...->g...", B, core5, Av, optimize="optimal")
    else:
        z1 = -Σ.prod() * H1e
        z2 = z2_scalar * I
        z3 = jnp.einsum("ig,ij,je->ge", B, adj, AH, optimize="optimal")
        z4 = jnp.einsum("ig,ij,je->ge", HB, adj, Aconj, optimize="optimal")
        z5 = jnp.einsum("ig,ij,je->ge", B, core5, Aconj, optimize="optimal")

    return -D * (z1 + z2 + z3 + z4 + z5)


def E2e_deriv(Aconj, B, *args):
    *v, H2e = args
    (v,) = v if len(v) > 0 else (None,)

    if H2e is None:
        return 0
    # d Aconj dB E2
    # where E2 is c^\dagger c
    no = B.shape[1]

    I = jnp.eye(no)

    _, _, Htt = antisymmetrized(H2e)

    M = Aconj @ B.T
    X, Σ, YH = jnp.linalg.svd(M)
    D = jnp.linalg.det(X) * jnp.linalg.det(YH)
    Xc, YHc = X.conj(), YH.conj()
    t1 = T2(Σ, 1)
    t1Xc = t1 * Xc
    t2 = T2(Σ, 2)
    t3 = T2(Σ, 3)
    U = B.T @ YHc.T
    W = Xc.T @ Aconj
    Htt_ket1 = jnp.einsum("abre,ce->abrc", Htt, B, optimize="optimal")
    Htt_ket2 = jnp.einsum("abge,cg,de->abcd", Htt, B, B, optimize="optimal")
    HttU = jnp.einsum("abrc,pc->abrp", Htt_ket1, YHc, optimize="optimal")
    HttUU = jnp.einsum("abcd,oc,pd->abop", Htt_ket2, YHc, YHc, optimize="optimal")
    z5 = 0
    for sgn, s in Sn("bar"):
        z5 -= 0.25 * sgn * jnp.einsum(f"abop,o{s[0]},p{s[1]},uq,q{s[2]},opq->ur", HttUU, W, W, U, W, t3, optimize="optimal")

    Htt_bra = jnp.einsum("aure,ja->jure", Htt, Aconj, optimize="optimal")
    HttUV = jnp.einsum("jure,ie->ijur", Htt_bra, B, optimize="optimal")

    if v is not None:
        z1 = jnp.einsum("jc,ci,ijur,r...->u...", t1Xc, YHc, HttUV, v, optimize="optimal")
        z2 =  jnp.einsum("abrp,uo,oa,pb,op,r...->u...", HttU, U, W, W, t2, v, optimize="optimal")
        z3 = - jnp.einsum("auop,or,pa,op,r...->u...", HttUU, W, W, t2, v, optimize="optimal")
        z4 = 0.5 *  jnp.einsum("abop,ob,pa,op->", HttUU, W, W, t2, optimize="optimal")
        z4 = z4 * v
        z5 = jnp.einsum("ur,r...->u...", z5, v, optimize="optimal")
    else:
        z1 =  jnp.einsum("jc,ci,ijur->ur", t1Xc, YHc, HttUV, optimize="optimal")
        z2 =  jnp.einsum("abrp,uo,oa,pb,op->ur", HttU, U, W, W, t2, optimize="optimal")
        z3 = - jnp.einsum("auop,or,pa,op->ur", HttUU, W, W, t2, optimize="optimal")
        z4 = 0.5 * jnp.einsum("abop,ob,pa,op->", HttUU, W, W, t2, optimize="optimal")
        z4 = z4 * I

    return D * (z1 + z2 + z3 + z4 + z5)
