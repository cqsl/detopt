import jax
import jax.numpy as jnp
from flax import struct


from .random import random_dets, normalize_det, randomly_rotate_orbitals
from .utils import normalize_orbs, orthogonalize_orbs

from sld.mels import compute_mels

from .base import SSDBASE

from .secondq import vslater_ansatz_index


@struct.dataclass
class GSSD(SSDBASE):
    X: jax.Array
    coeffs: jax.Array
    i: jax.Array = None  # orbital to derive wrt

    @classmethod
    def random(cls, key, ndet, nelec, n_orbitals, dtype=complex):
        X = random_dets(key, ndet, nelec, n_orbitals, dtype=dtype)
        coeffs = jnp.ones(ndet, dtype=X.dtype)
        return cls(X, coeffs)

    @property
    @staticmethod
    def ghf():
        return True

    @property
    def n_orbitals(self):
        return self.X.shape[-1]

    @property
    def nelec(self):
        return self.X.shape[-2]

    @property
    def ndet(self):
        return self.coeffs.shape[0]

    @property
    def dtype(self):
        return self.X.dtype

    def transform(self, U):
        X = jnp.einsum("...ei,ji->...ej" if U.ndim == 2 else "...ei,i->...ei", self.X, U)
        return self.replace(X=X)

    def thouless(self, T):
        return self.transform(jax.scipy.linalg.expm(T))

    def normalize_dets(self):
        X, csq = normalize_det(self.X)
        coeffs = self.coeffs * jnp.sqrt(csq)
        return self.replace(X=X, coeffs=coeffs)

    def normalize_orbs(self):
        X, c = normalize_orbs(self.X)
        coeffs = self.coeffs * c
        return self.replace(X=X, coeffs=coeffs)

    def norm(self, expval_impl, batch_size=None):
        S, _ = compute_mels(expval_impl, self, batch_size=batch_size)
        S = S[:, 0, :, 0]
        return jnp.sqrt(self.coeffs.conj() @ S @ self.coeffs)

    def normalize(self, expval_impl, batch_size=None):
        return self.replace(coeffs=self.coeffs / self.norm(expval_impl, batch_size))

    def secondq_eval(self, n):
        assert n.ndim == 2
        logpsi = vslater_ansatz_index(self.X, n)
        return self.coeffs @ jnp.exp(logpsi)

    def choose_random_orbitals_to_optimize(self, key, optimize_first_only=False):
        assert self.i is None
        if optimize_first_only:
            i = jnp.zeros(self.ndet, dtype=jnp.int32)
        else:
            i = jax.random.randint(key, minval=0, maxval=self.nelec, shape=self.ndet, dtype=jnp.int32)
        return self.replace(i=i)

    def clear_orbitals_to_optimize(self):
        return self.replace(i=None)

    def orthogonalize_orbitals(self):
        X = orthogonalize_orbs(self.X)
        return self.replace(X=X)

    def randomly_rotate_orbitals(self, key):
        X = randomly_rotate_orbitals(key, self.X)
        return self.replace(X=X)

    def set_orbitals(self, v0, i=None):
        if i is None:
            i = self.i
        assert i is not None
        X = jax.vmap(lambda X, i, v: X.at[i].set(v))(self.X, i, v0)
        return self.replace(X=X)

    def get_orbitals(self, i=None):
        if i is None:
            i = self.i
        assert i is not None
        return jax.vmap(lambda x, i: x[i])(self.X, i)

    def get_orbitals_weighted(self, i=None):
        return self.coeffs[:, None] * self.get_orbitals(i=i)

    def unfreeze(self, nfreeze):
        X = jnp.repeat(jnp.eye(self.n_orbitals + nfreeze, dtype=self.dtype)[None, :nfreeze, :], self.ndet, axis=0)
        X = jnp.concatenate([X, jnp.pad(self.X, ((0, 0), (0, 0), (nfreeze, 0)))], axis=1)
        return self.replace(X=X)

    @property
    def n_zero_eigvals(self):
        return (self.nelec - 1) * self.ndet

    # orbitals with the coefficients absorbed, as stored in the output files
    def _to_nonnormalized_arrays(self):
        return self.X * jnp.exp(jnp.log(self.coeffs)[:, None, None] / self.nelec)
