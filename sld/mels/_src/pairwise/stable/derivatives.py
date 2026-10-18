import jax.numpy as jnp


from .common import A_inv_det_stable_1, A_inv_det_stable_2, A_inv_det_stable_3


def ol_deriv(Uc, V, v=None):
    # dUc dV overlap
    ne, no = V.shape
    M = Uc @ V.T
    I = jnp.eye(no)
    detM, M_inv = A_inv_det_stable_1(M)
    if v is None:
        z = jnp.linalg.det(M) * I - detM * V.T @ (M_inv @ Uc)
    else:
        z = jnp.linalg.det(M) * v - detM * V.T @ (M_inv @ (Uc @ v))
    return z


def E1e_deriv(Uc, V, *args, mo_coeff=None):
    *v, H1e = args
    (v,) = v if len(v) > 0 else (None,)

    if H1e is None:
        return 0

    # dUc dV E1
    # where E1 is c^\dagger c (normal ordering)
    ne, no = V.shape

    M = Uc @ V.T
    I = jnp.eye(no)

    if mo_coeff is not None:
        # we need the full MO H1e anyway below
        H1e = jnp.einsum("AB,Aa,Bb->ab", H1e, mo_coeff, mo_coeff)

    detM, M_inv = A_inv_det_stable_1(M)
    D, MM = A_inv_det_stable_2(M)
    UHV = jnp.einsum("ib,ja,ab->ij", V, Uc, H1e, optimize="optimal")

    if v is not None:
        z1 = -jnp.linalg.det(M) * H1e @ v
        z2 = -detM * jnp.einsum("ij,ij", M_inv, UHV, optimize="optimal") * v
        z3 = detM * jnp.einsum("ig,ij,ja,ae,e...->g...", V, M_inv, Uc, H1e, v, optimize="optimal")
        z4 = detM * jnp.einsum("gb,ib,ij,je,e...->g...", H1e, V, M_inv, Uc, v, optimize="optimal")
        z5 = D * jnp.einsum("ijkl,ig,je,kl,e...->g...", MM, V, Uc, UHV, v, optimize="optimal")
    else:
        z1 = -jnp.linalg.det(M) * H1e
        UH = jnp.einsum("ja,ae->je", Uc, H1e, optimize="optimal")
        HV = jnp.einsum("gb,ib->ig", H1e, V, optimize="optimal")
        z2 = -detM * jnp.einsum("ij,ij", M_inv, UHV, optimize="optimal") * I
        z3 = detM * jnp.einsum("ig,ij,je->ge", V, M_inv, UH, optimize="optimal")
        z4 = detM * jnp.einsum("ig,ij,je->ge", HV, M_inv, Uc, optimize="optimal")
        z5 = D * jnp.einsum("ijkl,ig,je,kl->ge", MM, V, Uc, UHV, optimize="optimal")

    return -(z1 + z2 + z3 + z4 + z5)


def E2e_deriv(Uc, V, *args, mo_coeff=None, unroll=False):
    *v, H2e = args
    (v,) = v if len(v) > 0 else (None,)

    if H2e is None:
        return 0
    if unroll:
        raise NotImplementedError

    # dUc dV E2
    # where E2 is c^\dagger c
    ne, no = V.shape

    M = Uc @ V.T
    I = jnp.eye(no)

    if isinstance(H2e, tuple):
        H2e, Ht, Htt = H2e
        Ht2 = H2e + H2e.transpose(1, 0, 3, 2)
    else:
        Ht = H2e - H2e.transpose(0, 1, 3, 2)
        Ht2 = H2e + H2e.transpose(1, 0, 3, 2)
        Htt = Ht - Ht.transpose(1, 0, 2, 3)

    if mo_coeff is None:
        abge_fn = lambda abge: abge
        mo_coeff_arrays = ()
    else:
        mo_coeff_arrays = (mo_coeff,) * 4

        def abge_fn(abge):
            a, b, g, e = abge
            return f"ABGE,A{a},B{b},G{g},E{e}"

    detM, M_inv = A_inv_det_stable_1(M)
    D, MM = A_inv_det_stable_2(M)
    DD, MMM = A_inv_det_stable_3(M)

    Ht2VVU = jnp.einsum(f"{abge_fn('auge')},ie,kg,la->iklu", Ht2, *mo_coeff_arrays, V, V, Uc, optimize="optimal")
    HUUVV = jnp.einsum(f"{abge_fn('abge')},ka,lb,ig,je->ijkl", H2e, *mo_coeff_arrays, Uc, Uc, V, V, optimize="optimal")

    if v is not None:
        HttUVv = jnp.einsum(f"ie,ja,{abge_fn('aure')},r...->iju...", V, Uc, Htt, *mo_coeff_arrays, v, optimize="optimal")
        HtVUUv = jnp.einsum(f"{abge_fn('abre')},ke,ja,lb,r...->jkl...", Ht, *mo_coeff_arrays, V, Uc, Uc, v, optimize="optimal")
        z1 = detM * jnp.einsum("ij,iju...->u...", M_inv, HttUVv, optimize="optimal")
        z2 = D * jnp.einsum("ijkl,iu,jkl...->u...", MM, V, HtVUUv, optimize="optimal")
        z3 = D * jnp.einsum("ijkl,jr,iklu,r...->u...", MM, Uc, Ht2VVU, v, optimize="optimal")
        z4 = D * jnp.einsum("ijkl,iklj", MM, HUUVV, optimize="optimal") * v
        z5 = DD * jnp.einsum("ijklmn,nu,mr,ljki,r...->u...", MMM, V, Uc, HUUVV, v, optimize="optimal")
    else:
        HttUV = jnp.einsum(f"ie,ja,{abge_fn('aure')}->ijur", V, Uc, Htt, *mo_coeff_arrays, optimize="optimal")
        HtVUU = jnp.einsum(f"{abge_fn('abre')},ke,ja,lb->jklr", Ht, *mo_coeff_arrays, V, Uc, Uc, optimize="optimal")
        z1 = detM * jnp.einsum("ij,ijur->ur", M_inv, HttUV, optimize="optimal")
        z2 = D * jnp.einsum("ijkl,iu,jklr->ur", MM, V, HtVUU, optimize="optimal")
        z3 = D * jnp.einsum("ijkl,jr,iklu->ur", MM, Uc, Ht2VVU, optimize="optimal")
        z4 = D * jnp.einsum("ijkl,iklj", MM, HUUVV, optimize="optimal") * I
        z5 = DD * jnp.einsum("ijklmn,nu,mr,ljki->ur", MMM, V, Uc, HUUVV, optimize="optimal")

    return z1 + z2 + z3 + z4 + z5
