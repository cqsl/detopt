import jax
import jax.numpy as jnp

from sld.utils.hamiltonian.two_body import AntisymmetricTwoBody
from .common import hamiltonian_terms, red


def swap_if_needed(cond, Uconju_, Uconjd_, Vu_, Vd_, ku_, kd_, lu_, ld_):
    Vu = jax.lax.select(cond, Vd_, Vu_)
    Vd = jax.lax.select(cond, Vu_, Vd_)
    Uconju = jax.lax.select(cond, Uconjd_, Uconju_)
    Uconjd = jax.lax.select(cond, Uconju_, Uconjd_)

    # workaround for ShardingTypeError: select `which` must be scalar or have the same sharding as cases
    ku = cond * kd_ + (~cond) * ku_
    kd = cond * ku_ + (~cond) * kd_

    lu = jax.lax.select(cond, ld_, lu_)
    ld = jax.lax.select(cond, lu_, ld_)
    return Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld


def partially_contract_1e(rdm1_full_fn, Uconju, Vu, w, U_onsite=None):
    # H_lk = \sum_{ij} w_{ijkl} <Uu| c_j^\dagger c_i |Vu>
    if w is None and U_onsite is None:
        return None
    r = rdm1_full_fn(Uconju, Vu)
    if U_onsite is not None:
        return U_onsite * jnp.diag(r.diagonal())
    return jnp.einsum("ijkl,ij->lk", w, r)


def _hess_uu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, _swap=False):

    ol_d = impl.ol(Uconjd, Vd)
    dVdU_Ol_u = red(impl.ol_deriv, Uconju, ku, Vu, lu)
    Suu = dVdU_Ol_u * ol_d

    H1e, H2e, Enuc, w, U, U_onsite = hamiltonian_terms(hamiltonian)

    E1e_d = impl.E1e(Uconjd, Vd, H1e)
    E2e_d = impl.E2e(Uconjd, Vd, H2e)
    E12_d = E1e_d + E2e_d
    dVdU_E1e_u = red(impl.E1e_deriv, Uconju, ku, Vu, lu, H1e)
    dVdU_E2e_u = red(impl.E2e_deriv, Uconju, ku, Vu, lu, H2e)
    dVdU_E12_u = dVdU_E1e_u + dVdU_E2e_u
    tmp_uu = partially_contract_1e(impl.rdm1_full, Uconjd, Vd, w, U_onsite=U_onsite)
    Huu_w = red(impl.E1e_deriv, Uconju, ku, Vu, lu, tmp_uu)
    Huu = dVdU_E12_u * ol_d + dVdU_Ol_u * E12_d + Huu_w + Enuc * Suu

    if U is not None:
        Uu, Ud = U
        HuuP, _ = _hess_uu(impl, Uconju, Uconjd, Vu@Uu.T, Vd@Ud.T, ku, kd, lu, ld, None, _swap=_swap)
        U_active = jax.lax.select(_swap, Ud, Uu)
        HuuP = HuuP @ U_active
        Huu = Huu + HuuP
    return Suu, Huu


def _hess_dd(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian):
    return _hess_uu(impl, Uconjd, Uconju, Vd, Vu, kd, ku, ld, lu, hamiltonian, _swap=True)


def _hess_ud(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, _use_grad=False, _swap=False):
    if _use_grad:
        dU_Ol_u = red(impl.ol_deriv, Uconju, ku, Vu, lu, grad=True)
    else:
        Vu_luT = Vu.at[(lu,), :].get(out_sharding=jax.typeof(Vu).sharding).T
        dVdU_Ol_u = red(impl.ol_deriv, Uconju, ku, Vu, lu)
        dU_Ol_u = dVdU_Ol_u @ Vu_luT

    Uconjd_kd = Uconjd.at[(kd,), :].get(out_sharding=jax.typeof(Uconjd).sharding)
    dVdU_Ol_d = red(impl.ol_deriv, Uconjd, kd, Vd, ld)
    dV_Ol_d = Uconjd_kd @ dVdU_Ol_d

    Sud = dU_Ol_u * dV_Ol_d

    H1e, H2e, Enuc, w, U, U_onsite = hamiltonian_terms(hamiltonian)

    if _use_grad:
        dU_E1e_u = red(impl.E1e_deriv, Uconju, ku, Vu, lu, H1e, grad=True)
        dU_E2e_u = red(impl.E2e_deriv, Uconju, ku, Vu, lu, H2e, grad=True)
        dU_E12_u = dU_E1e_u + dU_E2e_u
    else:
        dVdU_E1e_u = red(impl.E1e_deriv, Uconju, ku, Vu, lu, H1e)
        dVdU_E2e_u = red(impl.E2e_deriv, Uconju, ku, Vu, lu, H2e)
        dVdU_E12_u = dVdU_E1e_u + dVdU_E2e_u
        if dVdU_E12_u.ndim > 0:
            dU_E12_u = dVdU_E12_u @ Vu_luT
        else:
            dU_E12_u = dVdU_E12_u  # = 0

    dVdU_E1e_d = red(impl.E1e_deriv, Uconjd, kd, Vd, ld, H1e)
    dVdU_E2e_d = red(impl.E2e_deriv, Uconjd, kd, Vd, ld, H2e)

    dVdU_E12_d = dVdU_E1e_d + dVdU_E2e_d
    if dVdU_E12_d.ndim > 0:
        dV_E12_d = Uconjd_kd @ dVdU_E12_d
    else:
        dV_E12_d = dVdU_E12_d  # = 0

    Hud_w = impl.interaction_ud_deriv(Uconju, ku, Vu, lu, Uconjd, kd, Vd, ld, w, swap=_swap, U_onsite=U_onsite)

    Hud = Enuc * Sud + dU_Ol_u * dV_E12_d + dU_E12_u * dV_Ol_d + Hud_w

    if U is not None:
        Uu, Ud = U
        HudP, _ = _hess_ud(impl, Uconju, Uconjd, Vu@Uu.T, Vd@Ud.T, ku, kd, lu, ld, None, _use_grad=_use_grad, _swap=_swap)
        if _swap:
            HudP = HudP@Uu
        else:
            HudP = HudP@Ud
        Hud = Hud + HudP
    return Sud, Hud


def _hess_du(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, _use_grad=False):
    return _hess_ud(impl, Uconjd, Uconju, Vd, Vu, kd, ku, ld, lu, hamiltonian, _use_grad=_use_grad, _swap=True)


def _can_use_direct_inverse_hessian(impl, hamiltonian):
    if not hasattr(impl, "hamiltonian_hess_same_spin_inverse"):
        return False
    if hamiltonian in [None, {}]:
        return True
    H2e = hamiltonian.get('H2e', None)
    H1e = hamiltonian.get('H1e', None)
    return (H2e is None or isinstance(H2e, AntisymmetricTwoBody)) and not isinstance(H1e, tuple) and hamiltonian.get('U', None) is None and hamiltonian.get('U_onsite', None) is None


def _direct_inverse_hessian_hamiltonian(hamiltonian):
    H1e, H2e, Enuc, w, _, _ = hamiltonian_terms(hamiltonian)
    return H1e, H2e, w, Enuc


def hess_spin_generic(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian):
    # UHF version

    s, t, k, l = deriv_indices

    ku, kd = k
    lu, ld = l

    if _can_use_direct_inverse_hessian(impl, hamiltonian):
        H1e, H2e, w, Enuc = _direct_inverse_hessian_hamiltonian(hamiltonian)
        Suu, Huu = impl.hamiltonian_hess_same_spin_inverse(Uconju, Uconjd, Vu, Vd, ku, lu, H1e, H2e, w, Enuc)
        Sud, Hud = impl.hamiltonian_hess_opposite_spin_inverse(Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, H1e, H2e, w, Enuc)
        Sdu, Hdu = impl.hamiltonian_hess_opposite_spin_inverse(Uconjd, Uconju, Vd, Vu, kd, ku, ld, lu, H1e, H2e, w, Enuc, swap=True)
        Sdd, Hdd = impl.hamiltonian_hess_same_spin_inverse(Uconjd, Uconju, Vd, Vu, kd, ld, H1e, H2e, w, Enuc)
    else:
        Suu, Huu = _hess_uu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)
        Sud, Hud = _hess_ud(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)
        Sdu, Hdu = _hess_du(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)
        Sdd, Hdd = _hess_dd(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)

    uu = (s == 0) * (t == 0)
    # ud = (s==0) * (t==1)
    du = (s == 1) * (t == 0)
    dd = (s == 1) * (t == 1)

    SS = jax.lax.select(uu, Suu, jax.lax.select(dd, Sdd, jax.lax.select(du, Sdu, Sud)))
    HH = jax.lax.select(uu, Huu, jax.lax.select(dd, Hdd, jax.lax.select(du, Hdu, Hud)))

    return SS, HH


def hess_spin_block(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, spin_block):
    _, _, (ku, kd), (lu, ld) = deriv_indices

    if _can_use_direct_inverse_hessian(impl, hamiltonian):
        H1e, H2e, w, Enuc = _direct_inverse_hessian_hamiltonian(hamiltonian)
        if spin_block == "uu":
            return impl.hamiltonian_hess_same_spin_inverse(Uconju, Uconjd, Vu, Vd, ku, lu, H1e, H2e, w, Enuc)
        if spin_block == "ud":
            return impl.hamiltonian_hess_opposite_spin_inverse(Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, H1e, H2e, w, Enuc)
        if spin_block == "du":
            return impl.hamiltonian_hess_opposite_spin_inverse(Uconjd, Uconju, Vd, Vu, kd, ku, ld, lu, H1e, H2e, w, Enuc, swap=True)
        if spin_block == "dd":
            return impl.hamiltonian_hess_same_spin_inverse(Uconjd, Uconju, Vd, Vu, kd, ld, H1e, H2e, w, Enuc)
    else:
        if spin_block == "uu":
            return _hess_uu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)
        if spin_block == "ud":
            return _hess_ud(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)
        if spin_block == "du":
            return _hess_du(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)
        if spin_block == "dd":
            return _hess_dd(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)

    raise ValueError(f"unknown UHF Hessian spin block: {spin_block!r}")


def hess_spin_sz0(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian):
    s, t, (ku, kd), (lu, ld) = deriv_indices
    dd_or_ud = t == 1
    uu_or_dd = s == t
    # there are 4 possible cases of spin of the lhs and rhs derivatives: uu ud du dd
    # here we swap the determinants so transforming dd->uu and ud->du so we only have to compute two cases instead of 4.
    Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld = swap_if_needed(dd_or_ud, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld)
    if _can_use_direct_inverse_hessian(impl, hamiltonian):
        H1e, H2e, w, Enuc = _direct_inverse_hessian_hamiltonian(hamiltonian)
        Suu, Huu = impl.hamiltonian_hess_same_spin_inverse(Uconju, Uconjd, Vu, Vd, ku, lu, H1e, H2e, w, Enuc)
        Sdu, Hdu = impl.hamiltonian_hess_opposite_spin_inverse(Uconjd, Uconju, Vd, Vu, kd, ku, ld, lu, H1e, H2e, w, Enuc, swap=True)
    else:
        # the swap exchanges the two spin sectors, which the w term has to know about
        Suu, Huu = _hess_uu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, _swap=dd_or_ud)
        Sdu, Hdu = _hess_du(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, _use_grad=True)
    SS = jax.lax.select(uu_or_dd, Suu, Sdu)
    HH = jax.lax.select(uu_or_dd, Huu, Hdu)
    return SS, HH


def hess_spin(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian):
    if Uconju.shape == Uconjd.shape:  # same number of electrons up and down
        return hess_spin_sz0(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian)
    else:
        return hess_spin_generic(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian)


def _gradu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian):
    ol_d = impl.ol(Uconjd, Vd)
    dU_Ol_u = red(impl.ol_deriv, Uconju, ku, Vu, lu, grad=True)
    Su = dU_Ol_u * ol_d

    H1e, H2e, Enuc, w, U, U_onsite = hamiltonian_terms(hamiltonian)

    E1e_d = impl.E1e(Uconjd, Vd, H1e)
    E2e_d = impl.E2e(Uconjd, Vd, H2e)
    E12_d = E1e_d + E2e_d
    dU_E1e_u = red(impl.E1e_deriv, Uconju, ku, Vu, lu, H1e, grad=True)
    dU_E2e_u = red(impl.E2e_deriv, Uconju, ku, Vu, lu, H2e, grad=True)
    dU_E12_u = dU_E1e_u + dU_E2e_u
    tmp_uu = partially_contract_1e(impl.rdm1_full, Uconjd, Vd, w, U_onsite=U_onsite)
    Hu_w = red(impl.E1e_deriv, Uconju, ku, Vu, lu, tmp_uu, grad=True)
    Hu = dU_E12_u * ol_d + dU_Ol_u * E12_d + Hu_w + Enuc * Su

    if U is not None:
        Uu, Ud = U
        # we only compute gradient wrt lhs. for the rhs the vector is already transformed with Uu / Ud
        HuP, _ = _gradu(impl, Uconju, Uconjd, Vu@Uu.T, Vd@Ud.T, ku, kd, lu, ld, None)
        Hu = Hu + HuP
    return Su, Hu


def _gradd(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian):
    return _gradu(impl, Uconjd, Uconju, Vd, Vu, kd, ku, ld, lu, hamiltonian)


def grad_spin_sz0(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian):
    s, _, (ku, kd), (lu, ld) = deriv_indices
    Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld = swap_if_needed(s == 1, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld)
    return _gradu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)


def grad_spin_generic(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian):
    s, _, (ku, kd), (lu, ld) = deriv_indices
    Su, Hu = _gradu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)
    Sd, Hd = _gradd(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian)
    SS = jax.lax.select(s == 0, Su, Sd)
    HH = jax.lax.select(s == 0, Hu, Hd)
    return SS, HH


def grad_spin(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian):
    s, t, k, l = deriv_indices
    # we need an arbitrary orbital to factor out for the rhs; if we don't get one choose the same as for the lhs
    if l is None:
        l = k
    deriv_indices = s, t, k, l

    if Uconju.shape == Uconjd.shape:  # same number of electrons up and down
        return grad_spin_sz0(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian)
    else:
        return grad_spin_generic(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian)
