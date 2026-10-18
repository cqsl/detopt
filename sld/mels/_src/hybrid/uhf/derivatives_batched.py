from functools import partial
import jax
import jax.numpy as jnp
from jax import P
from ..batching import _check_replicated, _check_sharding, _sector_first, _mesh, batch_map
from ..common import _check_degree, _unpack, barrier, metric, metric_bra, normalize_index, prepare_Htt, w_A, w_A_other, remove_row, split_spin
from .derivatives import _opposite_spin_pair, _same_spin_pair
from .mels import H_A


def _same_spin(A, Ao, B, Bo, bra_indices, ket_indices, Htt, w, H1, H1o, Enuc, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, S=None):
    def bra(data):
        A, Ao, bra_index = data
        bra_index = normalize_index(bra_index, A.shape[0])
        Ah = remove_row(A, bra_index)
        HA = H_A(Ah, Htt)
        HAo = H_A(Ao, Htt)
        Ah, HA = barrier(Ah, HA)
        Ao, HAo = barrier(Ao, HAo)
        wAo = w_A_other(Ao, w)
        AS = metric_bra(Ah, S)
        ASo = metric_bra(Ao, S)

        def ket(x):
            B, Bo, ket_index = x
            ket_index = normalize_index(ket_index, B.shape[0])
            sign = 1 - 2 * ((bra_index + ket_index) % 2)
            return _same_spin_pair(Ah, Ao, remove_row(B, ket_index), Bo, HA, HAo, wAo, H1, H1o, Enuc, sign, rcond, max_degree, factorization, S, AS, ASo)
        return batch_map(ket, (B, Bo, ket_indices), pair_batch_size)
    return batch_map(bra, (A, Ao, bra_indices), cached_batch_size)


def _opposite_spin(A, Ao, B, Bo, ias, ios, kas, kos, Htt, w, H1, H1o, Enuc, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, S=None):
    def bra(data):
        A, Ao, ia, io = data
        ia = normalize_index(ia, A.shape[0])
        io = normalize_index(io, Ao.shape[0])
        Ah = remove_row(A, ia)
        Aoh = remove_row(Ao, io)
        HA = H_A(Ah, Htt)
        HAo = H_A(Aoh, Htt)
        Ah, HA = barrier(Ah, HA)
        Aoh, HAo = barrier(Aoh, HAo)
        wAo = w_A_other(Aoh, w)
        wA = w_A(Ah, w)
        AS = metric_bra(Ah, S)
        ASo = metric_bra(Aoh, S)

        def ket(x):
            B, Bo, ka, ko = x
            ka = normalize_index(ka, B.shape[0])
            ko = normalize_index(ko, Bo.shape[0])
            sign = 1 - 2 * ((ia + ka + io + ko) % 2)
            return _opposite_spin_pair(Ah, Aoh, remove_row(B, ka), remove_row(Bo, ko), B[ka], Ao[io], HA, HAo, wAo, wA, w, H1, H1o, Enuc, sign, rcond, max_degree, factorization, S, AS, ASo)
        return batch_map(ket, (B, Bo, kas, kos), pair_batch_size)
    return batch_map(bra, (A, Ao, ias, ios), cached_batch_size)


def hessian_spin_block(Aconju, Aconjd, Bu, Bd, bra_indices, ket_indices, H1e=None, H2e=None, w=None, Enuc=0, spin_block='uu', rcond=1e-10, max_degree=4, factorization='svd', cached_batch_size=16, pair_batch_size=128, S=None):
    max_degree = _check_degree(max_degree, 4)
    H1u, H1d = split_spin(H1e)
    Htt = prepare_Htt(H2e, Aconju, Aconjd, Bu, Bd)
    bra_u, bra_d = bra_indices
    ket_u, ket_d = ket_indices
    args = (rcond, max_degree, factorization, cached_batch_size, pair_batch_size, S)
    if spin_block == 'uu':
        return _same_spin(Aconju, Aconjd, Bu, Bd, bra_u, ket_u, Htt, w, H1u, H1d, Enuc, *args)
    if spin_block == 'dd':
        return _same_spin(Aconjd, Aconju, Bd, Bu, bra_d, ket_d, Htt, w, H1d, H1u, Enuc, *args)
    if spin_block == 'ud':
        return _opposite_spin(Aconju, Aconjd, Bu, Bd, bra_u, bra_d, ket_u, ket_d, Htt, w, H1u, H1d, Enuc, *args)
    if spin_block == 'du':
        return _opposite_spin(Aconjd, Aconju, Bd, Bu, bra_d, bra_u, ket_d, ket_u, Htt, w, H1d, H1u, Enuc, *args)
    raise ValueError("spin_block must be 'uu', 'ud', 'du', or 'dd'")


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def hessian(X, arrays=None, rcond=1e-10, max_degree=4, factorization='svd', cached_batch_size=16, pair_batch_size=128):
    if X.s is None or X.i is None:
        raise ValueError('the hybrid UHF Hessian requires selected UHF determinants')
    H1e, H2e, w, Enuc = _unpack(arrays, 'UHF hessian')
    S = metric(arrays)
    nup = (X.ndet + 1) // 2
    Xc = X.conj()
    Xcu, Xcd, Yu, Yd = (Xc[:nup], Xc[nup:], X[:nup], X[nup:])

    def evaluate(A, B, spin_block):
        return hessian_spin_block(A.Xu, A.Xd, B.Xu, B.Xd, (A.i[:, 0], A.i[:, 1]), (B.i[:, 0], B.i[:, 1]), H1e, H2e, w, Enuc, spin_block, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, S)
    Suu, Huu = evaluate(Xcu, Yu, 'uu')
    Sud, Hud = evaluate(Xcu, Yd, 'ud')
    Sdu, Hdu = evaluate(Xcd, Yu, 'du')
    Sdd, Hdd = evaluate(Xcd, Yd, 'dd')

    def assemble(uu, ud, du, dd):
        return jnp.concatenate((jnp.concatenate((uu, ud), axis=1), jnp.concatenate((du, dd), axis=1)), axis=0).transpose(0, 2, 1, 3)
    return assemble(Suu, Sud, Sdu, Sdd), assemble(Huu, Hud, Hdu, Hdd)


def hessian_row_block(up, Aconju, Aconjd, bra_indices, Yu, Yd, H1e=None, H2e=None, w=None, Enuc=0, S=None, rcond=1e-10, max_degree=4, factorization='svd', cached_batch_size=16, pair_batch_size=128):
    max_degree = _check_degree(max_degree, 4)
    H1u, H1d = split_spin(H1e)
    A = _sector_first(up, (Aconju, Aconjd))
    Ai = _sector_first(up, (bra_indices[:, 0], bra_indices[:, 1]))
    H1 = (H1u, H1d) if H1u is H1d else _sector_first(up, (H1u, H1d))
    same = _sector_first(up, Yu[:2], Yd[:2])
    same_indices = _sector_first(up, (Yu[2][:, 0], Yu[2][:, 1]), (Yd[2][:, 0], Yd[2][:, 1]))
    opposite = _sector_first(up, Yd[:2], Yu[:2])
    opposite_indices = _sector_first(up, (Yd[2][:, 0], Yd[2][:, 1]), (Yu[2][:, 0], Yu[2][:, 1]))

    def block(B, ket_indices, spin_block):
        return hessian_spin_block(*A, *B, Ai, ket_indices, H1, H2e, w, Enuc, spin_block, rcond, max_degree, factorization, cached_batch_size, pair_batch_size, S)
    Ssame, Hsame = block(same, same_indices, 'uu')
    Sopposite, Hopposite = block(opposite, opposite_indices, 'ud')

    def place(same_block, opposite_block):
        return jnp.concatenate(_sector_first(up, (same_block, opposite_block)), axis=1)
    return place(Ssame, Sopposite), place(Hsame, Hopposite)


@partial(jax.jit, static_argnames=('max_degree', 'factorization', 'cached_batch_size', 'pair_batch_size'))
def hessian_sharded(X, arrays=None, rcond=1e-10, max_degree=4, factorization='svd', cached_batch_size=16, pair_batch_size=128, Y=None):
    mesh, nshards = _mesh()
    if nshards == 1:
        return hessian(X, arrays, rcond, max_degree, factorization, cached_batch_size, pair_batch_size)
    if Y is None:
        raise ValueError('Y is required on multiple devices so the caller can provide its sharding')
    if X.s is None or X.i is None or Y.s is None or Y.i is None:
        raise ValueError('the hybrid UHF Hessian requires selected UHF determinants')
    if X.ndet != Y.ndet:
        raise ValueError('the sharded UHF Hessian requires equally sized bra and ket batches')
    if X.nelec[0] != X.nelec[1]:
        raise ValueError('the sharded UHF Hessian requires equal spin electron counts')
    if X.ndet % nshards != 0:
        raise ValueError('the sharded UHF Hessian requires the shard count to divide the determinants')
    if nshards % 2 != 0:
        raise ValueError('the bra-sharded UHF Hessian requires an even shard count')
    H1e, H2e, w, Enuc = _unpack(arrays, 'UHF hessian')
    hamiltonian = _check_replicated(H1e, H2e, w, Enuc, metric(arrays))

    _check_sharding(X, P('i'), 'X')
    _check_sharding(Y, P(), 'Y')
    nup = X.ndet // 2
    Xc = X.conj()
    halves = ((Y.Xu[:nup], Y.Xd[:nup], Y.i[:nup]), (Y.Xu[nup:], Y.Xd[nup:], Y.i[nup:]))

    def row_block(Xu, Xd, bra_indices, Yu, Yd, hamiltonian):
        up = jax.lax.axis_index('i') < nshards // 2
        return hessian_row_block(up, Xu, Xd, bra_indices, Yu, Yd, *hamiltonian, rcond, max_degree, factorization, cached_batch_size, pair_batch_size)

    evaluate = jax.shard_map(
        row_block,
        mesh=mesh,
        in_specs=(P('i'), P('i'), P('i'), P(), P(), P()),
        out_specs=(P('i'), P('i')),
    )
    overlap, value = evaluate(Xc.Xu, Xc.Xd, Xc.i, *halves, hamiltonian)
    return tuple(block.transpose(0, 2, 1, 3) for block in (overlap, value))
