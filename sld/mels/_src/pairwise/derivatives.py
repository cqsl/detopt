from .common import hamiltonian_terms, red


def hess(impl, Uconj, V, deriv_indices=None, hamiltonian=None):
    # GHF version

    k, l = deriv_indices

    H1e, H2e, Enuc, _, U, _ = hamiltonian_terms(hamiltonian)
    if U is not None:
        raise NotImplementedError("U operators in the GHF Hessian")

    SS = red(impl.ol_deriv, Uconj, k, V, l)

    dVdU_E1e = red(impl.E1e_deriv, Uconj, k, V, l, H1e)
    dVdU_E2e = red(impl.E2e_deriv, Uconj, k, V, l, H2e)
    HH = dVdU_E1e + dVdU_E2e + Enuc * SS
    return SS, HH
