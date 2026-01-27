import jax
import jax.numpy as jnp

from .mels import rdm1_red

from sld.utils.sparse import einsum_maybe_sparse


def interaction_ud_deriv(Aconju, iu, Bu, ku, Aconjd, kd, Bd, jd, w, mo_coeffu=None, mo_coeffd=None, swap=False):
    # compute \partial_{ui}^\dagger \partial_{dj} <A|H|B>
    if w is None:
        return 0

    # to use the hessian matrix elements we need symmetric equations. remove an arbitrary orbital here
    # ku and kd are arbitrary

    no = Aconju.shape[1]
    I = jnp.eye(no)

    Buku = Bu.at[ku].get()
    Bu = jnp.delete(Bu, ku, axis=0, assume_unique_indices=True)
    Aconju = jnp.delete(Aconju, iu, axis=0, assume_unique_indices=True)
    Mu = Aconju @ Bu.T
    r1u = rdm1_red(Aconju, Bu)

    Adconjkd = Aconjd.at[kd].get(out_sharding=jax.typeof(Aconjd).sharding)
    Bd = jnp.delete(Bd, jd, axis=0, assume_unique_indices=True)
    Aconjd = jnp.delete(Aconjd, kd, axis=0, assume_unique_indices=True)
    Md = Aconjd @ Bd.T
    r1d = rdm1_red(Aconjd, Bd)

    # we integrate again
    a00u = Buku - jnp.einsum("ga,a->g", r1u, Buku)
    a00d = Adconjkd - jnp.einsum("be,b->e", r1d, Adconjkd)

    res = 0

    if mo_coeffu is not None:
        r1uao = jnp.einsum("ab,Aa,Bb->AB", r1u, mo_coeffu, mo_coeffu)
        I_m_r1u_ao = jnp.einsum("Aa,ia->iA", mo_coeffu, I - r1u)
        a00u_ao = jnp.einsum("Bb,b->B", mo_coeffu, a00u)
    else:
        r1uao = r1u
        I_m_r1u_ao = I - r1u
        a00u_ao = a00u

    if mo_coeffd is not None:
        r1dao = jnp.einsum("ab,Aa,Bb->AB", r1d, mo_coeffd, mo_coeffd)
        I_m_r1d_ao = jnp.einsum("Ee,ej->Ej", mo_coeffd, I - r1d)
        a00d_ao = jnp.einsum("Gg,g->G", mo_coeffd, a00d)
    else:
        r1dao = r1d
        I_m_r1d_ao = I - r1d
        a00d_ao = a00d

    if swap:
        abge = "GEAB"
    else:
        abge = "ABGE"

    tmp0 = einsum_maybe_sparse(f"{abge},BA,EG", w, r1uao, r1dao, notation="chemist")
    tmp1a = einsum_maybe_sparse(f"{abge},EG->AB", w, r1dao, notation="chemist")
    tmp2a = einsum_maybe_sparse(f"{abge},BA,G->E", w, r1uao, a00d_ao, notation="chemist")
    tmp3a = einsum_maybe_sparse(f"{abge},B,G->AE", w, a00u_ao, a00d_ao, notation="chemist")

    tmp1 = jnp.einsum("AB,iA,B->i", tmp1a, I_m_r1u_ao, a00u_ao)
    tmp2 = jnp.einsum("E,Ej->j", tmp2a, I_m_r1d_ao)
    tmp3 = jnp.einsum("AE,iA,Ej->ij", tmp3a, I_m_r1u_ao, I_m_r1d_ao)

    res += jnp.einsum("i,j->ij", a00u, a00d) * tmp0
    res += jnp.einsum("i,j->ij", tmp1, a00d)
    res += jnp.einsum("i,j->ij", a00u, tmp2)
    res += tmp3

    assert jnp.issubdtype(ku.dtype, jnp.signedinteger)
    sign = 1 - 2 * ((iu + ku + kd + jd) % 2)  # (-1) ** (iu + ku + kd + jd)

    res = sign * jnp.linalg.det(Mu) * jnp.linalg.det(Md) * res
    return res
