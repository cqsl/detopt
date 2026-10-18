import jax
import jax.numpy as jnp

from ..common import r1h_M_r1h, r1h_v, v_r1h
from .mels import inv_and_det, rdm1_red_from_inverse

from sld.utils.hamiltonian.two_body import AntisymmetricTwoBody


def _rdm1(Aconj, B):
    Minv, det = inv_and_det(Aconj @ B.T)
    L = B.T @ Minv
    return L @ Aconj, det, L


def _rdm1_minor(Aconj, i, B, k):
    # remove orbital i of the bra and k of the ket
    i = jnp.where(i < 0, i + Aconj.shape[0], i)
    k = jnp.where(k < 0, k + B.shape[0], k)
    Ac = jnp.delete(Aconj, i, axis=0, assume_unique_indices=True)
    B = jnp.delete(B, k, axis=0, assume_unique_indices=True)
    r1, det, L = _rdm1(Ac, B)
    r1h = jnp.eye(Aconj.shape[1], dtype=Aconj.dtype) - r1
    sign = 1 - 2 * ((i + k) % 2)
    return r1, r1h, sign * det, L, Ac


def _energy_H1eff(r1, H1e, H2e):
    e = jnp.asarray(0, dtype=r1.dtype)
    H1eff = jnp.zeros_like(r1)
    if H1e is not None:
        e = e + jnp.einsum("ba,ab", r1, H1e)
        H1eff = H1eff + H1e
    if H2e is not None:
        if not isinstance(H2e, AntisymmetricTwoBody):
            raise NotImplementedError("the unstable UHF hessian needs an AntisymmetricTwoBody H2e")
        H1eff_2e = jnp.einsum("EA,ABGE->BG", r1, 4 * H2e.Ha, optimize=True)
        e = e + 0.5 * jnp.einsum("GB,BG", r1, H1eff_2e)
        H1eff = H1eff + H1eff_2e
    return e, H1eff


def hamiltonian_hess_same_spin_inverse(Aconju, Aconjd, Bu, Bd, i, k, H1e, H2e, w, Enuc):
    r1u, r1hu, detu, Lu, Acu = _rdm1_minor(Aconju, i, Bu, k)
    r1d, detd, _ = _rdm1(Aconjd, Bd)
    eu, H1eff_u = _energy_H1eff(r1u, H1e, H2e)
    ed, _ = _energy_H1eff(r1d, H1e, H2e)

    eud = jnp.asarray(0, dtype=r1u.dtype)
    if w is not None:
        Hw_u = jnp.einsum("ijkl,ij->lk", w, r1d)
        eud = jnp.einsum("ba,ab", r1u, Hw_u)
        H1eff_u = H1eff_u + Hw_u

    det = detu * detd
    S = det * r1hu
    H = det * ((Enuc + eu + ed + eud) * r1hu + r1h_M_r1h(H1eff_u, Lu, Acu, Lu, Acu))
    return S, H


def hamiltonian_hess_opposite_spin_inverse(Aconju, Aconjd, Bu, Bd, iu, kd, ku, jd, H1e, H2e, w, Enuc, swap=False):
    r1u, _, detu, Lu, Acu = _rdm1_minor(Aconju, iu, Bu, ku)
    r1d, _, detd, Ld, Acd = _rdm1_minor(Aconjd, kd, Bd, jd)
    Buk = Bu.at[(ku,), :].get(out_sharding=jax.typeof(Bu).sharding).reshape(Bu.shape[1])
    Adk = Aconjd.at[(kd,), :].get(out_sharding=jax.typeof(Aconjd).sharding).reshape(Aconjd.shape[1])
    a00u = r1h_v(Lu, Acu, Buk)
    a00d = v_r1h(Adk, Ld, Acd)
    a00ud = jnp.einsum("i,j->ij", a00u, a00d)

    eu, H1eff_u = _energy_H1eff(r1u, H1e, H2e)
    ed, H1eff_d = _energy_H1eff(r1d, H1e, H2e)
    res = (Enuc + eu + ed) * a00ud
    res += jnp.einsum("i,j->ij", r1h_v(Lu, Acu, H1eff_u @ a00u), a00d)
    res += jnp.einsum("i,j->ij", a00u, v_r1h(a00d @ H1eff_d, Ld, Acd))

    if w is not None:
        abge = "GEAB" if swap else "ABGE"
        tmp1a = jnp.einsum(f"{abge},EG->AB", w, r1d)
        tmp0 = jnp.einsum("AB,BA", tmp1a, r1u)
        tmp2a = jnp.einsum(f"{abge},BA,G->E", w, r1u, a00d)
        tmp3a = jnp.einsum(f"{abge},B,G->AE", w, a00u, a00d)
        res += tmp0 * a00ud
        res += jnp.einsum("i,j->ij", r1h_v(Lu, Acu, tmp1a @ a00u), a00d)
        res += jnp.einsum("i,j->ij", a00u, v_r1h(tmp2a, Ld, Acd))
        res += r1h_M_r1h(tmp3a, Lu, Acu, Ld, Acd)

    det = detu * detd
    return det * a00ud, det * res


def interaction_ud_deriv(Aconju, iu, Bu, ku, Aconjd, kd, Bd, jd, w, swap=False, U_onsite=None):
    # compute \partial_{ui}^\dagger \partial_{dj} <A|H|B>
    if w is None and U_onsite is None:
        return 0

    # to use the hessian matrix elements we need symmetric equations; ku and kd are arbitrary orbitals removed here
    I = jnp.eye(Aconju.shape[1])

    Buku = Bu.at[ku].get()
    Bu = jnp.delete(Bu, ku, axis=0, assume_unique_indices=True)
    Aconju = jnp.delete(Aconju, iu, axis=0, assume_unique_indices=True)
    Mu_inv, det_Mu = inv_and_det(Aconju @ Bu.T)
    r1u = rdm1_red_from_inverse(Aconju, Bu, Mu_inv)

    Adconjkd = Aconjd.at[kd].get(out_sharding=jax.typeof(Aconjd).sharding)
    Bd = jnp.delete(Bd, jd, axis=0, assume_unique_indices=True)
    Aconjd = jnp.delete(Aconjd, kd, axis=0, assume_unique_indices=True)
    Md_inv, det_Md = inv_and_det(Aconjd @ Bd.T)
    r1d = rdm1_red_from_inverse(Aconjd, Bd, Md_inv)

    a00u = Buku - jnp.einsum("ga,a->g", r1u, Buku)
    a00d = Adconjkd - jnp.einsum("be,b->e", r1d, Adconjkd)

    assert jnp.issubdtype(ku.dtype, jnp.signedinteger)
    sign = 1 - 2 * ((iu + ku + kd + jd) % 2)

    if U_onsite is not None:
        res = jnp.einsum("i,aa,j,aa->ij", a00u, r1u, a00d, r1d)
        res += jnp.einsum("ia,a,j,aa->ij", I - r1u, a00u, a00d, r1d)
        res += jnp.einsum("i,aa,a,aj->ij", a00u, r1u, a00d, I - r1d)
        res += jnp.einsum("ia,a,a,aj->ij", I - r1u, a00u, a00d, I - r1d)
        return sign * det_Mu * det_Md * res * U_onsite

    abge = "GEAB" if swap else "ABGE"
    tmp1a = jnp.einsum(f"{abge},EG->AB", w, r1d)
    tmp0 = jnp.einsum("AB,BA", tmp1a, r1u)
    tmp2a = jnp.einsum(f"{abge},BA,G->E", w, r1u, a00d)
    tmp3a = jnp.einsum(f"{abge},B,G->AE", w, a00u, a00d)

    tmp1 = jnp.einsum("AB,iA,B->i", tmp1a, I - r1u, a00u)
    tmp2 = jnp.einsum("E,Ej->j", tmp2a, I - r1d)
    tmp3 = jnp.einsum("AE,iA,Ej->ij", tmp3a, I - r1u, I - r1d)

    res = jnp.einsum("i,j->ij", a00u, a00d) * tmp0
    res += jnp.einsum("i,j->ij", tmp1, a00d)
    res += jnp.einsum("i,j->ij", a00u, tmp2)
    res += tmp3
    return sign * det_Mu * det_Md * res
