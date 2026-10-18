import jax.numpy as jnp
from ..algebra import contract, outer, coeff_at
from ...pairwise.common import r1h_M_r1h, r1h_v, v_r1h
from ..common import metric_bra, metric_factor, projector
from .mels import _cross_contractions, _sector_energy, _total_energy


def _sector(Aconj, B, Htt_bra, H1e, rcond, max_degree, factorization, S=None, AS=None):
    AS = metric_bra(Aconj, S) if AS is None else AS
    L, K, det, null, energy, dtype = _sector_energy(Aconj, B, Htt_bra, H1e, rcond, max_degree, factorization, AS)
    LS = metric_factor(L, S)
    Q = projector(LS, AS, dtype, S)
    no = Aconj.shape[1]
    operator = jnp.zeros((no, no), dtype=dtype) if K is None else K
    if H1e is not None:
        operator = operator + H1e
    return L, LS, Q, det, null, energy, operator


def _same_spin_pair(A, Ao, B, Bo, HA, HAo, wAo, H1, H1o, Enuc, sign, rcond, max_degree, factorization, S=None, AS=None, ASo=None):
    AS = metric_bra(A, S) if AS is None else AS
    ASo = metric_bra(Ao, S) if ASo is None else ASo
    L, LS, Q, deta, nulla, ea, op = _sector(A, B, HA, H1, rcond, max_degree, factorization, S, AS)
    Lo, _, _, deto, nullo, eo, _ = _sector(Ao, Bo, HAo, H1o, rcond, max_degree, factorization, S, ASo)
    cross = None
    if wAo is not None:
        cross, _ = _cross_contractions(L, Lo, wAo, None)
        op = op + cross
    energy = _total_energy(A, L, ea, eo, cross, Enuc)
    hamiltonian = energy * Q + r1h_M_r1h(op, LS, A, L, AS)
    det = sign * deta * deto
    null = None if nulla is None or nullo is None else nulla + nullo
    return det * coeff_at(Q, null), det * coeff_at(hamiltonian, null)


def _opposite_spin_pair(A, Ao, B, Bo, Bk, Ak, HA, HAo, wAo, wA, w, H1, H1o, Enuc, sign, rcond, max_degree, factorization, S=None, AS=None, ASo=None):
    AS = metric_bra(A, S) if AS is None else AS
    ASo = metric_bra(Ao, S) if ASo is None else ASo
    L, LS, Q, deta, nulla, ea, op = _sector(A, B, HA, H1, rcond, max_degree, factorization, S, AS)
    Lo, LSo, Qo, deto, nullo, eo, opo = _sector(Ao, Bo, HAo, H1o, rcond, max_degree, factorization, S, ASo)
    cross = None
    if w is not None:
        cross, crosso = _cross_contractions(L, Lo, wAo, wA)
        op = op + cross
        opo = opo + crosso
    energy = _total_energy(A, L, ea, eo, cross, Enuc)
    q = r1h_v(L, AS, Bk)
    p = v_r1h(Ak, LSo, Ao)
    q_free = q if S is None else S @ q
    p_free = p if S is None else p @ S
    qp = outer(q_free, p_free)
    left = r1h_v(LS, A, op @ q)
    right = v_r1h(p @ opo, Lo, ASo)
    hamiltonian = energy * qp + outer(left, p_free) + outer(q_free, right)
    if w is not None:
        mixed = contract('ABGE,B,G->AE', w, q, p)
        hamiltonian = hamiltonian + r1h_M_r1h(mixed, LS, A, Lo, ASo)
    det = sign * deta * deto
    null = None if nulla is None or nullo is None else nulla + nullo
    return det * coeff_at(qp, null), det * coeff_at(hamiltonian, null)
