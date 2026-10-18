import jax
import jax.numpy as jnp


from sld.utils.sparse import einsum_maybe_sparse
from .common import red


def swap_if_needed(cond, Uconju_, Uconjd_, Vu_, Vd_, ku_, kd_, lu_, ld_, mo_coeffu_, mo_coeffd_):
    Vu = jax.lax.select(cond, Vd_, Vu_)
    Vd = jax.lax.select(cond, Vu_, Vd_)
    Uconju = jax.lax.select(cond, Uconjd_, Uconju_)
    Uconjd = jax.lax.select(cond, Uconju_, Uconjd_)

    # workaround for ShardingTypeError: select `which` must be scalar or have the same sharding as cases
    # ku = jax.lax.select(cond, kd_, ku_)
    # kd = jax.lax.select(cond, ku_, kd_)
    ku = cond * kd_ + (~cond) * ku_
    kd = cond * ku_ + (~cond) * kd_

    lu = jax.lax.select(cond, ld_, lu_)
    ld = jax.lax.select(cond, lu_, ld_)
    if mo_coeffu_ is not None:
        assert mo_coeffd_ is not None
        mo_coeffu = jax.lax.select(cond, mo_coeffd_, mo_coeffu_)
        mo_coeffd = jax.lax.select(cond, mo_coeffu_, mo_coeffd_)
    else:
        assert mo_coeffu_ is None
        mo_coeffu = mo_coeffd = None
    return Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, mo_coeffu, mo_coeffd


def partially_contract_1e(rdm1_full_fn, Uconju, Vu, w, mo_coeff=None, swap=True):
    # H_d_kl = < Uu | \sum_{ij} w_{ijkl} c_i^\dagger c_j c_k^\dagger |Vu>
    if w is None:
        return None
    r = rdm1_full_fn(Uconju, Vu)
    if mo_coeff is not None:
        r = jnp.einsum("eg,Ee,Gg->EG", r, mo_coeff, mo_coeff)
    if not swap:
        res = einsum_maybe_sparse("ABGE,BA->GE", w, r, notation="chemist")
    else:
        res = einsum_maybe_sparse("ABGE,EG->AB", w, r, notation="chemist")
    return res


def hess(impl, Uconj, V, deriv_indices=None, hamiltonian=None, mo_coeff=None, unroll=False):
    # version withuot spin  / GHF version

    k, l = deriv_indices

    if hamiltonian is not None:
        H2e, H1e, Enuc = hamiltonian
    else:
        H2e, H1e, Enuc = None, None, 1

    SS = red(impl.ol_deriv, Uconj, k, V, l)

    dVdU_E1e = red(impl.E1e_deriv, Uconj, k, V, l, H1e, mo_coeff=mo_coeff)
    dVdU_E2e = red(impl.E2e_deriv, Uconj, k, V, l, H2e, mo_coeff=mo_coeff, unroll=unroll)
    HH = dVdU_E1e + dVdU_E2e + Enuc * SS
    return SS, HH


def _hess_uu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=False, _swap=False):
    # for dd swap the arguments (pass u stuff for d and vice versa and set swap=True if w is not symmetric)

    ol_d = impl.ol(Uconjd, Vd)
    dVdU_Ol_u = red(impl.ol_deriv, Uconju, ku, Vu, lu)
    Suu = dVdU_Ol_u * ol_d

    if hamiltonian is None:
        H2e, H1e, Enuc, w = None, None, 1, None
    else:
        H2e, H1e, Enuc, w = hamiltonian

    E1e_d = impl.E1e(Uconjd, Vd, H1e, mo_coeff=mo_coeffd)
    E2e_d = impl.E2e(Uconjd, Vd, H2e, mo_coeff=mo_coeffd)
    E12_d = E1e_d + E2e_d
    dVdU_E1e_u = red(impl.E1e_deriv, Uconju, ku, Vu, lu, H1e, mo_coeff=mo_coeffu)
    dVdU_E2e_u = red(impl.E2e_deriv, Uconju, ku, Vu, lu, H2e, mo_coeff=mo_coeffu, unroll=unroll)
    dVdU_E12_u = dVdU_E1e_u + dVdU_E2e_u
    # this is on down, so invert the swap
    tmp_uu = partially_contract_1e(impl.rdm1_full, Uconjd, Vd, w, mo_coeff=mo_coeffd, swap=not _swap)
    Huu_w = red(impl.E1e_deriv, Uconju, ku, Vu, lu, tmp_uu, mo_coeff=mo_coeffu)
    Huu = dVdU_E12_u * ol_d + dVdU_Ol_u * E12_d + Huu_w + Enuc * Suu
    return Suu, Huu


def _hess_dd(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=False):
    return _hess_uu(impl, Uconjd, Uconju, Vd, Vu, kd, ku, ld, lu, hamiltonian, mo_coeffd, mo_coeffu, unroll=unroll, _swap=True)


def _hess_ud(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=False, _use_grad=False, _swap=False):
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

    if hamiltonian is None:
        H2e, H1e, Enuc, w = None, None, 1, None
    else:
        H2e, H1e, Enuc, w = hamiltonian

    if _use_grad:
        dU_E1e_u = red(impl.E1e_deriv, Uconju, ku, Vu, lu, H1e, mo_coeff=mo_coeffu, grad=True)
        dU_E2e_u = red(impl.E2e_deriv, Uconju, ku, Vu, lu, H2e, mo_coeff=mo_coeffu, unroll=unroll, grad=True)
        dU_E12_u = dU_E1e_u + dU_E2e_u
    else:
        dVdU_E1e_u = red(impl.E1e_deriv, Uconju, ku, Vu, lu, H1e, mo_coeff=mo_coeffu)
        dVdU_E2e_u = red(impl.E2e_deriv, Uconju, ku, Vu, lu, H2e, mo_coeff=mo_coeffu, unroll=unroll)
        dVdU_E12_u = dVdU_E1e_u + dVdU_E2e_u
        dU_E12_u = dVdU_E12_u @ Vu_luT

    dVdU_E1e_d = red(impl.E1e_deriv, Uconjd, kd, Vd, ld, H1e, mo_coeff=mo_coeffd)
    dVdU_E2e_d = red(impl.E2e_deriv, Uconjd, kd, Vd, ld, H2e, mo_coeff=mo_coeffd, unroll=unroll)

    dVdU_E12_d = dVdU_E1e_d + dVdU_E2e_d
    if dVdU_E12_d.ndim > 0:
        dV_E12_d = Uconjd_kd @ dVdU_E12_d
    else:
        dV_E12_d = dVdU_E12_d  # = 0

    Hud_w = impl.interaction_ud_deriv(Uconju, ku, Vu, lu, Uconjd, kd, Vd, ld, w, mo_coeffu, mo_coeffd, swap=_swap)

    Hud = Enuc * Sud + dU_Ol_u * dV_E12_d + dU_E12_u * dV_Ol_d + Hud_w

    return Sud, Hud


def _hess_du(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=False, _use_grad=False):
    return _hess_ud(impl, Uconjd, Uconju, Vd, Vu, kd, ku, ld, lu, hamiltonian, mo_coeffd, mo_coeffu, unroll=unroll, _use_grad=_use_grad, _swap=True)


def hess_spin_generic(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeffu, mo_coeffd, unroll=False):
    # UHF version

    s, t, k, l = deriv_indices

    ku, kd = k
    lu, ld = l

    Suu, Huu = _hess_uu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)
    Sud, Hud = _hess_ud(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)
    Sdu, Hdu = _hess_du(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)
    Sdd, Hdd = _hess_dd(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)

    uu = (s == 0) * (t == 0)
    # ud = (s==0) * (t==1)
    du = (s == 1) * (t == 0)
    dd = (s == 1) * (t == 1)

    SS = jax.lax.select(uu, Suu, jax.lax.select(dd, Sdd, jax.lax.select(du, Sdu, Sud)))
    HH = jax.lax.select(uu, Huu, jax.lax.select(dd, Hdd, jax.lax.select(du, Hdu, Hud)))

    return SS, HH


def hess_spin_sz0(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeffu, mo_coeffd, unroll=False):
    s, t, (ku, kd), (lu, ld) = deriv_indices
    dd_or_ud = t == 1
    uu_or_dd = s == t
    # there are 4 possible cases of spin of the lhs and rhs derivatives: uu ud du dd
    # here we swap the determinants so transforming dd->uu and ud->du so we only have to compute two cases instead of 4.
    Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, mo_coeffu, mo_coeffd = swap_if_needed(dd_or_ud, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, mo_coeffu, mo_coeffd)
    Suu, Huu = _hess_uu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)
    Sdu, Hdu = _hess_du(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll, _use_grad=True)
    SS = jax.lax.select(uu_or_dd, Suu, Sdu)
    HH = jax.lax.select(uu_or_dd, Huu, Hdu)
    return SS, HH


def hess_spin(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeff=None, unroll=False):
    if mo_coeff is None:
        mo_coeffu = mo_coeffd = None
    elif isinstance(mo_coeff, tuple):
        mo_coeffu, mo_coeffd = mo_coeff
    else:
        mo_coeffu = mo_coeffd = mo_coeff

    if Uconju.shape == Uconjd.shape:  # same number of electrons up and down
        return hess_spin_sz0(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)
    else:
        # TODO figure out if we can somehow pad to the max nelec and use a more efficient code like the one for Sz=0
        return hess_spin_generic(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)


def _gradu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu=None, mo_coeffd=None, unroll=False, _swap=False):
    # for gradd swap the arguments (pass u stuff for d and vice versa and set swap=True if w is not symmetric)
    ol_d = impl.ol(Uconjd, Vd)
    dU_Ol_u = red(impl.ol_deriv, Uconju, ku, Vu, lu, grad=True)
    Su = dU_Ol_u * ol_d

    if hamiltonian is None:
        H2e, H1e, Enuc, w = None, None, 1, None
    else:
        H2e, H1e, Enuc, w = hamiltonian

    E1e_d = impl.E1e(Uconjd, Vd, H1e, mo_coeff=mo_coeffd)
    E2e_d = impl.E2e(Uconjd, Vd, H2e, mo_coeff=mo_coeffd)
    E12_d = E1e_d + E2e_d
    dU_E1e_u = red(impl.E1e_deriv, Uconju, ku, Vu, lu, H1e, mo_coeff=mo_coeffu, grad=True)
    dU_E2e_u = red(impl.E2e_deriv, Uconju, ku, Vu, lu, H2e, mo_coeff=mo_coeffu, unroll=unroll, grad=True)
    dU_E12_u = dU_E1e_u + dU_E2e_u
    # this is on down, so invert the swap
    tmp_uu = partially_contract_1e(impl.rdm1_full, Uconjd, Vd, w, mo_coeff=mo_coeffd, swap=not _swap)
    Hu_w = red(impl.E1e_deriv, Uconju, ku, Vu, lu, tmp_uu, grad=True, mo_coeff=mo_coeffu)
    Hu = dU_E12_u * ol_d + dU_Ol_u * E12_d + Hu_w + Enuc * Su
    return Su, Hu


def _gradd(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu=None, mo_coeffd=None, unroll=False):
    return _gradu(impl, Uconjd, Uconju, Vd, Vu, kd, ku, ld, lu, hamiltonian, mo_coeffd, mo_coeffu, unroll=unroll, _swap=True)


def grad_spin_sz0(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeffu, mo_coeffd, unroll=False):
    s, _, (ku, kd), (lu, ld) = deriv_indices
    Su, Hu = _gradu(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)
    Sd, Hd = _gradd(impl, Uconju, Uconjd, Vu, Vd, ku, kd, lu, ld, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)
    SS = jax.lax.select(s == 0, Su, Sd)
    HH = jax.lax.select(s == 0, Hu, Hd)
    return SS, HH


def grad_spin_generic(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeffu, mo_coeffd, unroll):
    raise NotImplementedError


def grad_spin(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeff=None, unroll=False):
    if mo_coeff is None:
        mo_coeffu = mo_coeffd = None
    elif isinstance(mo_coeff, tuple):
        mo_coeffu, mo_coeffd = mo_coeff
    else:
        mo_coeffu = mo_coeffd = mo_coeff

    s, t, k, l = deriv_indices
    # we need an arbitrary orbital to factor out for the rhs; if we don't get one choose the same as for the lhs
    if l is None:
        l = k
    deriv_indices = s, t, k, l

    if Uconju.shape == Uconjd.shape:  # same number of electrons up and down
        return grad_spin_sz0(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)
    else:
        # TODO figure out if we can somehow pad to the max nelec and use a more efficient code like the one for Sz=0
        return grad_spin_generic(impl, Uconju, Uconjd, Vu, Vd, deriv_indices, hamiltonian, mo_coeffu, mo_coeffd, unroll=unroll)
