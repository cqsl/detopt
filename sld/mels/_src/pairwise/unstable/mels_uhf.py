import jax.numpy as jnp

from sld.utils.hamiltonian.two_body import AntisymmetricTwoBody

from ..common import ol
from .mels import rdm1_red, rdm1_red_and_det, E1e_red, E2e_red


def _direct_expval_spin(X1uc, X1dc, X2u, X2d, H1e, H2e, w, Enuc):
    r1u, olu = rdm1_red_and_det(X1uc, X2u)
    r1d, old = rdm1_red_and_det(X1dc, X2d)
    H1e_u, H1e_d = H1e if isinstance(H1e, tuple) else (H1e, H1e)
    energy = jnp.asarray(Enuc, dtype=r1u.dtype)
    if H1e_u is not None:
        energy = energy + jnp.einsum("ba,ab", r1u, H1e_u)
    if H1e_d is not None:
        energy = energy + jnp.einsum("ba,ab", r1d, H1e_d)
    if H2e is not None:
        H1eff_u = jnp.einsum("EA,ABGE->BG", r1u, 4 * H2e.Ha, optimize=True)
        H1eff_d = jnp.einsum("EA,ABGE->BG", r1d, 4 * H2e.Ha, optimize=True)
        energy = energy + 0.5 * jnp.einsum("GB,BG", r1u, H1eff_u) + 0.5 * jnp.einsum("GB,BG", r1d, H1eff_d)
    if w is not None:
        energy = energy + jnp.einsum("ijkl,ij,kl", w, r1d, r1u)
    overlap = olu * old
    return overlap[None, None], (overlap * energy)[None, None]


def E2eud_red(X1uc, X1dc, X2u, X2d, w):
    if w is None:
        return 0
    r1u = rdm1_red(X1uc, X2u)
    r1d = rdm1_red(X1dc, X2d)

    return jnp.einsum("ijkl,ij,kl", w, r1d, r1u)


def Eonsite_ud_red(X1uc, X1dc, X2u, X2d, U_onsite, r1u=None, r1d=None):
    if r1u is None:
        r1u = rdm1_red(X1uc, X2u)
    if r1d is None:
        r1d = rdm1_red(X1dc, X2d)
    return U_onsite * jnp.einsum("ii,ii", r1d, r1u)


def expval_spin(X1uc, X1dc, X2u, X2d, deriv_indices=None, hamiltonian=None):
    # compute <>, <H>
    # UHF version;
    # non-svd version; assumes overlap is nonzero

    if hamiltonian in [None, {}]:
        ol_ = ol(X1uc, X2u) * ol(X1dc, X2d)
        return ol_[None, None], ol_[None, None]

    H2e = hamiltonian.get('H2e', None)
    H1e = hamiltonian.get('H1e', None)
    Enuc = hamiltonian.get('Enuc', 0)
    w = hamiltonian.get('w', None)
    U = hamiltonian.get('U', None)
    U_onsite = hamiltonian.get('U_onsite', None)

    if U is None and U_onsite is None and (H2e is None or isinstance(H2e, AntisymmetricTwoBody)):
        return _direct_expval_spin(X1uc, X1dc, X2u, X2d, H1e, H2e, w, Enuc)

    ol_ = ol(X1uc, X2u) * ol(X1dc, X2d)

    if U_onsite is not None:
        if H2e is not None or w is not None or U is not None:
            raise NotImplementedError("U_onsite cannot be combined with H2e, w, or U")

        eu1_ = E1e_red(X1uc, X2u, H1e)
        ed1_ = E1e_red(X1dc, X2d, H1e)
        eud_ = Eonsite_ud_red(X1uc, X1dc, X2u, X2d, U_onsite)
        return ol_[None, None], (ol_ * (Enuc + ed1_ + eu1_ + eud_))[None, None]

    if isinstance(H1e, tuple):
        H1e_u, H1e_d = H1e
    else:
        H1e_u = H1e_d = H1e

    eu1_ = E1e_red(X1uc, X2u, H1e_u)
    ed1_ = E1e_red(X1dc, X2d, H1e_d)
    eu2_ = E2e_red(X1uc, X2u, H2e)
    ed2_ = E2e_red(X1dc, X2d, H2e)
    eud_ = E2eud_red(X1uc, X1dc, X2u, X2d, w)
    expval = ol_ * (Enuc + ed1_ + ed2_ + eu1_ + eu2_ + eud_)

    if U is not None:
        Uu, Ud = U
        expval = expval + ol(X1uc, X2u@Uu.T) * ol(X1dc, X2d@Uu.T)

    # match derivative interface for use in optimization
    return ol_[None, None], expval[None, None]


def expval_sq_spin(X1uc, X1dc, X2u, X2d, deriv_indices=None, hamiltonian=None):
    raise NotImplementedError("<H^2> has no unstable implementation; use the stable or hybrid one")
