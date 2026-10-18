import jax.numpy as jnp
from ..algebra import result_type, coeff_at, trace
from ...pairwise.common import r1h_v
from ..common import _check_degree, _factors, metric_bra, metric_factor, normalize_index, remove_row


def grad_rectangular_prepare_A(Aconj, bra_index, layouts):
    Ah = remove_row(Aconj, bra_index)
    return Ah, None if layouts is None else layouts.prepare(Ah)


def grad_rectangular_prepare_B(B, ket_index, layouts):
    Bh = remove_row(B, ket_index)
    return Bh, None if layouts is None else layouts.prepare(Bh)


def _grad_rectangular_project(Ah, Bh, L, det, null, Bk, bra_index, ket_index, K, H1e, H1e_extra, same_spin_scale, S=None, AS=None):
    AS = Ah if AS is None else AS
    H1 = None if H1e is None else same_spin_scale * H1e
    if H1e_extra is not None:
        H1 = H1e_extra if H1 is None else H1 + H1e_extra
    op = same_spin_scale * K
    energy = same_spin_scale * 0.5 * trace(Ah @ K @ L)
    if H1 is not None:
        op = op + H1
        energy = energy + trace(Ah @ H1 @ L)
    defect = r1h_v(L, AS, Bk)
    projected = r1h_v(metric_factor(L, S), Ah, op @ defect)
    if S is not None:
        defect = S @ defect
    hamiltonian = energy * defect + projected
    sign = 1 - 2 * ((normalize_index(bra_index, Ah.shape[0] + 1) + normalize_index(ket_index, Bh.shape[0] + 1)) % 2)
    det = sign * det
    return det * coeff_at(defect, null), det * coeff_at(hamiltonian, null)


def _zero_K(Ah, dtype):
    no = Ah.shape[1]
    return jnp.zeros((no, no), dtype=dtype)


def grad_rectangular_A_first_pair(Ah, B, bra_index, ket_index, layouts, prepared, H1e, H1e_extra=None, same_spin_scale=1, rcond=0, max_degree=3, factorization='svd', S=None, AS=None):
    max_degree = _check_degree(max_degree, 3)
    ket_index = normalize_index(ket_index, B.shape[0])
    Bh = remove_row(B, ket_index)
    AS = metric_bra(Ah, S) if AS is None else AS
    _, L, det, null = _factors(Ah, Bh, rcond, max_degree, factorization, AS)
    dtype = result_type(Ah, Bh)
    K = _zero_K(Ah, dtype) if prepared is None else layouts.contract(L, prepared)
    return _grad_rectangular_project(Ah, Bh, L, det, null, B[ket_index], bra_index, ket_index, K, H1e, H1e_extra, same_spin_scale, S, AS)


def grad_rectangular_B_first_pair(Aconj, Bh, bra_index, ket_index, layouts, prepared, Bk, H1e, H1e_extra=None, same_spin_scale=1, rcond=0, max_degree=3, factorization='svd', S=None):
    max_degree = _check_degree(max_degree, 3)
    bra_index = normalize_index(bra_index, Aconj.shape[0])
    Ah = remove_row(Aconj, bra_index)
    AS = metric_bra(Ah, S)
    P, L, det, null = _factors(Ah, Bh, rcond, max_degree, factorization, AS)
    dtype = result_type(Ah, Bh)
    K = _zero_K(Ah, dtype) if prepared is None else layouts.contract(P @ Ah, prepared)
    return _grad_rectangular_project(Ah, Bh, L, det, null, Bk, bra_index, ket_index, K, H1e, H1e_extra, same_spin_scale, S, AS)
