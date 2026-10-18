from ..algebra import coeff_at
from ...pairwise.common import r1h_v
from ..common import metric_bra, normalize_index, remove_row
from .derivatives import _sector
from .mels import _cross_contractions, _total_energy


def _gradient_pair(A, Ao, B, Bo, ket_index, HA, HAo, wAo, H1, H1o, Enuc, sign, rcond, max_degree, factorization, S=None, AS=None, ASo=None):
    ket_index = normalize_index(ket_index, B.shape[0])
    AS = metric_bra(A, S) if AS is None else AS
    ASo = metric_bra(Ao, S) if ASo is None else ASo
    L, LS, _, deta, nulla, ea, op = _sector(A, remove_row(B, ket_index), HA, H1, rcond, max_degree, factorization, S, AS)
    Lo, _, _, deto, nullo, eo, _ = _sector(Ao, Bo, HAo, H1o, rcond, max_degree, factorization, S, ASo)
    cross = None
    if wAo is not None:
        cross, _ = _cross_contractions(L, Lo, wAo, None)
        op = op + cross
    energy = _total_energy(A, L, ea, eo, cross, Enuc)
    defect = r1h_v(L, AS, B[ket_index])
    projected = r1h_v(LS, A, op @ defect)
    if S is not None:
        defect = S @ defect
    hamiltonian = energy * defect + projected
    det = sign * deta * deto
    null = None if nulla is None or nullo is None else nulla + nullo
    return (det * coeff_at(defect, null))[..., None], (det * coeff_at(hamiltonian, null))[..., None]
