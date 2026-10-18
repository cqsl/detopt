import jax
import jax.numpy as jnp


class SSDBASE:
    # let np use our operators (e.g. __rmul__) instead of converting to an array
    __array_ufunc__ = None

    def __len__(self):
        return self.ndet

    def __add__(self, other):
        if other == 0:
            return self
        assert isinstance(other, type(self))
        return jax.tree.map(lambda *a: jnp.concatenate(a, axis=0), self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, c):
        return self.replace(coeffs=c * self.coeffs)

    def __rmul__(self, c):
        return self.__mul__(c)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self.replace(coeffs=-self.coeffs)

    def __getitem__(self, *args):
        # select subsets of determinants
        return jax.tree.map(lambda x: x.__getitem__(*args), self)

    def conj(self):
        return jax.tree.map(lambda x: x.conj(), self)

    def pad(self, key, npad):
        return self + type(self).random(key, npad, self.nelec, self.n_orbitals, dtype=complex)

    @classmethod
    def empty(cls, nelec, n_orbitals, dtype=complex):
        fake_key = jax.random.key(0)
        return cls.random(fake_key, 0, nelec, n_orbitals, dtype=dtype)
