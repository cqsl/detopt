import jax.numpy as jnp
from ..algebra import result_type, coeff_at, trace, zeros_like
from ..common import _check_degree, _factors, _uses_enuc, metric_bra, metric_factor, projector


def _E2e_derivative(Aconj, L, K, S=None, AS=None, LS=None):
    dtype = result_type(Aconj, L, K)
    identity = jnp.eye(Aconj.shape[1], dtype=dtype) if S is None else S
    AS = Aconj if AS is None else AS
    LS = L if LS is None else LS
    AK = Aconj @ K
    AKL = AK @ L
    energy = 0.5 * trace(AKL)
    small = AKL @ AS - AK - energy * AS
    return LS @ small - K @ L @ AS + K + energy * identity


def E2e_deriv_A_first_pair(Aconj, B, layouts, prepared, rcond=0, max_degree=3, factorization='svd', S=None, AS=None):
    max_degree = _check_degree(max_degree, 3)
    AS = metric_bra(Aconj, S) if AS is None else AS
    _, L, det, null = _factors(Aconj, B, rcond, max_degree, factorization, AS)
    K = layouts.contract(L, prepared)
    return det * coeff_at(_E2e_derivative(Aconj, L, K, S, AS, metric_factor(L, S)), null)


def E2e_deriv_B_first_pair(Aconj, B, layouts, prepared, rcond=0, max_degree=3, factorization='svd', S=None, AS=None):
    max_degree = _check_degree(max_degree, 3)
    AS = metric_bra(Aconj, S) if AS is None else AS
    P, L, det, null = _factors(Aconj, B, rcond, max_degree, factorization, AS)
    K = layouts.contract(P @ Aconj, prepared)
    return det * coeff_at(_E2e_derivative(Aconj, L, K, S, AS, metric_factor(L, S)), null)


def ol_deriv_pair(Aconj, B, rcond=0, max_degree=1, factorization='svd', S=None, AS=None):
    max_degree = _check_degree(max_degree, 1)
    AS = metric_bra(Aconj, S) if AS is None else AS
    _, L, det, null = _factors(Aconj, B, rcond, max_degree, factorization, AS)
    Q = projector(metric_factor(L, S), AS, result_type(Aconj, B), S)
    return det * coeff_at(Q, null)


def E1e_prepare_A(Aconj, H1e):
    return Aconj @ H1e


def E1e_prepare_B(B, H1e):
    return H1e @ B.T


def _E1e_derivative(Aconj, L, P, AH, HBt, H1e, S=None, AS=None, LS=None):
    dtype = result_type(Aconj, L, H1e)
    identity = jnp.eye(Aconj.shape[1], dtype=dtype) if S is None else S
    AS = Aconj if AS is None else AS
    LS = L if LS is None else LS
    HL = HBt @ P
    AHL = AH @ L
    energy = trace(AHL)
    small = -AH + AHL @ AS - energy * AS
    return LS @ small + H1e - HL @ AS + energy * identity


def E1e_deriv_pair(Aconj, B, AH, HBt, H1e, rcond=0, max_degree=2, factorization='svd', S=None, AS=None):
    max_degree = _check_degree(max_degree, 2)
    AS = metric_bra(Aconj, S) if AS is None else AS
    P, L, det, null = _factors(Aconj, B, rcond, max_degree, factorization, AS)
    return det * coeff_at(_E1e_derivative(Aconj, L, P, AH, HBt, H1e, S, AS, metric_factor(L, S)), null)


def _hess_pair(Aconj, B, has_two_body, AH, HBt, H1e, Enuc, K_from, rcond, max_degree, factorization, S=None, AS=None):
    AS = metric_bra(Aconj, S) if AS is None else AS
    P, L, det, null = _factors(Aconj, B, rcond, max_degree, factorization, AS)
    LS = metric_factor(L, S)
    Q = projector(LS, AS, result_type(Aconj, B), S)
    hamiltonian = zeros_like(Q)
    if _uses_enuc(Enuc):
        hamiltonian = hamiltonian + Enuc * Q
    if H1e is not None:
        hamiltonian = hamiltonian + _E1e_derivative(Aconj, L, P, AH, HBt, H1e, S, AS, LS)
    if has_two_body:
        hamiltonian = hamiltonian + _E2e_derivative(Aconj, L, K_from(P, L), S, AS, LS)
    return det * coeff_at(Q, null), det * coeff_at(hamiltonian, null)
