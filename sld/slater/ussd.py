import jax
import jax.numpy as jnp
from flax import struct


from .random import random_dets, randomly_rotate_orbitals
from .utils import normalize_det, orthogonalize_orbs, normalize_orbs, sele1, sele2
from sld.mels import compute_mels

from .base import SSDBASE
from .gssd import GSSD

from .secondq import vslater_ansatz_index_split_spin


@struct.dataclass
class USSD(SSDBASE):
    Xu: jax.Array
    Xd: jax.Array
    coeffs: jax.Array
    s: jax.Array = None  # spin to derive wrt
    i: jax.Array = None  # orbital to derive wrt

    @classmethod
    def random(cls, key, ndet, nelec, n_orbitals, dtype=complex):
        k0, k1 = jax.random.split(key)
        Xu = random_dets(k0, ndet, nelec[0], n_orbitals, dtype=dtype)
        Xd = random_dets(k1, ndet, nelec[1], n_orbitals, dtype=dtype)
        coeffs = jnp.ones(ndet, dtype=Xu.dtype)
        return cls(Xu, Xd, coeffs)

    @property
    @staticmethod
    def ghf():
        return False

    @property
    def n_orbitals(self):
        return self.Xu.shape[-1]

    @property
    def nelec(self):
        return self.Xu.shape[-2], self.Xd.shape[-2]

    @property
    def ndet(self):
        return self.coeffs.shape[0]

    @property
    def dtype(self):
        return self.Xu.dtype

    @property
    def n_zero_eigvals(self):
        if self.nelec[0] == self.nelec[1]:
            return (self.nelec[0] - 1) * self.ndet
        else:
            return jnp.where(self.s == 0, self.nelec[0] - 1, self.nelec[1] - 1).sum()

    def transform(self, Uu, Ud):
        Xu = jnp.einsum("...ei,ji->...ej" if Uu.ndim == 2 else "...ei,i->...ei", self.Xu, Uu)
        Xd = jnp.einsum("...ei,ji->...ej" if Ud.ndim == 2 else "...ei,i->...ei", self.Xd, Ud)
        return self.replace(Xu=Xu, Xd=Xd)

    def thouless(self, Tu, Td):
        return self.transform(jax.scipy.linalg.expm(Tu) if Tu.ndim == 2 else jnp.exp(Tu), jax.scipy.linalg.expm(Td) if Td.ndim == 2 else jnp.exp(Td))

    def normalize_dets(self):
        Xu, cu = normalize_det(self.Xu)
        Xd, cd = normalize_det(self.Xd)
        coeffs = self.coeffs * jnp.sqrt(cu * cd)
        return self.replace(Xu=Xu, Xd=Xd, coeffs=coeffs)

    def normalize_orbs(self):
        Xu, cu = normalize_orbs(self.Xu)
        Xd, cd = normalize_orbs(self.Xd)
        coeffs = self.coeffs * cu * cd
        return self.replace(Xu=Xu, Xd=Xd, coeffs=coeffs)

    def norm(self, expval_impl, batch_size=None):
        S, _ = compute_mels(expval_impl, self, batch_size=batch_size)
        S = S[:, 0, :, 0]
        return jnp.sqrt(self.coeffs.conj() @ S @ self.coeffs)

    def normalize(self, expval_impl, batch_size=None):
        return self.replace(coeffs=self.coeffs / self.norm(expval_impl, batch_size))

    def secondq_eval(self, n):
        assert n.ndim == 2
        logpsi = vslater_ansatz_index_split_spin(self.Xu, self.Xd, n)
        return self.coeffs @ jnp.exp(logpsi)

    def to_generalized(self, i=None, s=None):
        # conversion to GHF, needed for some of the symmetry projections
        if s is None:
            s = self.s
        if i is None:
            i = self.i

        X = jax.vmap(lambda Xu, Xd: jnp.block([[Xu, jnp.zeros_like(Xu)], [jnp.zeros_like(Xd), Xd]]))(self.Xu, self.Xd)

        if i is not None:
            assert s is not None
            ku, kd = i.T
            ne_u, _ = self.nelec
            i = (1 - s) * ku + s * (kd + ne_u)
        return GSSD(X, self.coeffs, i)

    def choose_random_orbitals_to_optimize(self, key, optimize_first_only=False):
        assert self.i is None
        assert self.s is None
        k1, k2, k3 = jax.random.split(key, 3)
        s = jax.random.randint(k1, minval=0, maxval=2, shape=self.ndet, dtype=jnp.int32)
        if optimize_first_only:
            i = jnp.zeros((self.ndet, 2), dtype=jnp.int32)
        else:
            iu = jax.random.randint(k2, minval=0, maxval=self.nelec[0], shape=self.ndet, dtype=jnp.int32)
            id = jax.random.randint(k3, minval=0, maxval=self.nelec[1], shape=self.ndet, dtype=jnp.int32)
            i = jnp.array([iu, id]).T
        return self.replace(s=s, i=i)

    def clear_orbitals_to_optimize(self):
        return self.replace(s=None, i=None)

    def orthogonalize_orbitals(self, s=None):
        if s is None:
            s = self.s
        assert s is not None
        Xu = orthogonalize_orbs(self.Xu)
        Xd = orthogonalize_orbs(self.Xd)
        Xu, Xd = sele2(s, (Xu, Xd), (self.Xu, self.Xd))
        return self.replace(Xu=Xu, Xd=Xd)

    def randomly_rotate_orbitals(self, key, i=None, s=None):
        if s is None:
            s = self.s
        assert s is not None
        Xu, Xd = sele2(s, randomly_rotate_orbitals(key, (self.Xu, self.Xd)), (self.Xu, self.Xd))
        return self.replace(Xu=Xu, Xd=Xd)

    def set_orbitals(self, v0, i=None, s=None):
        if s is None:
            s = self.s
        if i is None:
            i = self.i
        assert i is not None
        assert s is not None
        tmp = jax.vmap(lambda X, i, v: X.at[i].set(v))(self.Xu, i.T[0], v0), jax.vmap(lambda X, i, v: X.at[i].set(v))(self.Xd, i.T[1], v0)
        Xu, Xd = sele2(s, tmp, (self.Xu, self.Xd))
        return self.replace(Xu=Xu, Xd=Xd)

    def get_orbitals(self, s=None, i=None):
        if s is None:
            s = self.s
        if i is None:
            i = self.i
        assert i is not None
        assert s is not None
        x0 = jax.vmap(lambda x, i: x[i])(self.Xu, i.T[0]), jax.vmap(lambda x, i: x[i])(self.Xd, i.T[1])
        x0 = sele1(s, x0)
        return x0

    def get_orbitals_weighted(self, s=None, i=None):
        return self.coeffs[:, None] * self.get_orbitals(s=s, i=i)

    def unfreeze(self, nfreezeu, nfreezed):
        Xu = jnp.repeat(jnp.eye(self.n_orbitals + nfreezeu, dtype=self.dtype)[None, :nfreezeu, :], self.ndet, axis=0)
        Xu = jnp.concatenate([Xu, jnp.pad(self.Xu, ((0, 0), (0, 0), (nfreezeu, 0)))], axis=1)
        Xd = jnp.repeat(jnp.eye(self.n_orbitals + nfreezed, dtype=self.dtype)[None, :nfreezed, :], self.ndet, axis=0)
        Xd = jnp.concatenate([Xd, jnp.pad(self.Xd, ((0, 0), (0, 0), (nfreezed, 0)))], axis=1)
        return self.replace(Xu=Xu, Xd=Xd)

    # orbitals with the coefficients absorbed, as stored in the output files
    def _to_nonnormalized_arrays(self):
        return jax.tree.map(lambda X: X * jnp.exp(jnp.log(self.coeffs)[:, None, None] / sum(self.nelec)), (self.Xu, self.Xd))
