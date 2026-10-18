from itertools import product
import jax.numpy as jnp
from flax import struct
from sld.utils.hamiltonian.two_body import AntisymmetricTwoBody
from ..pairwise.common import antisymmetrized
from .algebra import block, concatenate, contract, result_type
from .common import reorder_Htt

_SPIN_BLOCKS = ((0, 1, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 0), (1, 1, 0, 1, 0, 1), (2, -1, 1, 0, 0, 1), (2, -1, 0, 1, 1, 0))
_PREPARED_BLOCKS = ((0, 1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1, 0), (1, 1, 1, 0, 1, 0, 0), (1, 1, 0, 1, 0, 1, 0), (1, -1, 1, 0, 0, 1, 1), (1, -1, 0, 1, 1, 0, 1))


def is_keepspin(H2e):
    return isinstance(H2e, tuple) and len(H2e) == 2


def keepspin_coefficients(H2e):
    Huuuu, Hudud = H2e
    Hudud = jnp.asarray(Hudud)
    return antisymmetrized(Huuuu)[2], Hudud, Hudud.transpose(0, 1, 3, 2)


def two_body_dtype(H2e, *arrays):
    if H2e is None:
        return result_type(*arrays)
    parts = H2e if is_keepspin(H2e) else (H2e,)
    return result_type(*arrays, *(p.Ha if isinstance(p, AntisymmetricTwoBody) else p for p in parts))


@struct.dataclass
class TwoBodyLayout:
    layouts: tuple
    side: str = struct.field(pytree_node=False)
    nmo: int = struct.field(pytree_node=False)
    _PREPARE = {'bra': 'aebg,ka->kebg', 'ket': 'eabg,ke->kabg'}
    _CONTRACT = {'bra': ('ek,kebg->bg', 'ek,kgbe->bg'), 'ket': ('ka,kabg->bg', 'ka,kbag->bg')}

    @property
    def keepspin(self):
        return self.nmo is not None

    def _half(self, spin):
        return slice(0, self.nmo) if spin == 0 else slice(self.nmo, 2 * self.nmo)

    def prepare(self, X):
        subscript = self._PREPARE[self.side]
        if not self.keepspin:
            return contract(subscript, self.layouts[0], X),
        return tuple(contract(subscript, self.layouts[index // 2], X[:, self._half(index % 2)]) for index in range(2 * len(self.layouts)))

    def contract(self, factor, prepared):
        spellings = self._CONTRACT[self.side]
        if not self.keepspin:
            return contract(spellings[0], factor, prepared[0])
        blocks = {}
        for layout, sign, sa, sb, sg, se, variant in _PREPARED_BLOCKS:
            contracted, free = (sa, se) if self.side == 'bra' else (se, sa)
            half = self._half(free)
            part = factor[half] if self.side == 'bra' else factor[:, half]
            term = contract(spellings[variant], part, prepared[2 * layout + contracted])
            if sign < 0:
                term = -term
            blocks[sb, sg] = term if (sb, sg) not in blocks else blocks[sb, sg] + term
        return block([[blocks[0, 0], blocks[0, 1]], [blocks[1, 0], blocks[1, 1]]])


def two_body(H2e, side, dtype):
    if H2e is None:
        return None
    if side not in ('bra', 'ket'):
        raise ValueError("side must be 'bra' or 'ket'")
    axis = 1 if side == 'bra' else 0
    if is_keepspin(H2e):
        arrays = keepspin_coefficients(H2e)[:2]
        layouts = tuple(jnp.moveaxis(array, 3, axis).astype(dtype) for array in arrays)
        return TwoBodyLayout(layouts, side, arrays[0].shape[0])
    return TwoBodyLayout((reorder_Htt(antisymmetrized(H2e)[2], axis, dtype),), side, None)


@struct.dataclass
class KeepSpinTwoBody:
    arrays: tuple


def _spin_sliced(operand, letters, spaces, spin, nmo):
    half = lambda s: slice(0, nmo) if s == 0 else slice(nmo, 2 * nmo)
    index = tuple(half(spin[letter]) if kind == 'o' and letter in spin else slice(None) for letter, kind in zip(letters, spaces))
    if all((part == slice(None) for part in index)):
        return operand
    return operand[index]


def keepspin_einsum(subscripts, spaces, operands, nmo):
    inputs = subscripts.split('->')[0].split(',')
    spaces = spaces.split(',')
    positions = [i for i, operand in enumerate(operands) if isinstance(operand, KeepSpinTwoBody)]
    expansion = []
    for choice in product(_SPIN_BLOCKS, repeat=len(positions)):
        spin, sign, consistent = ({}, 1, True)
        for position, block in zip(positions, choice):
            layout, block_sign = (block[0], block[1])
            for letter, value in zip(inputs[position], block[2:]):
                if spin.setdefault(letter, value) != value:
                    consistent = False
                    break
            if not consistent:
                break
            sign *= block_sign
        if not consistent:
            continue
        blocks = dict(zip(positions, choice))
        args = [operands[i].arrays[blocks[i][0]] if i in blocks else _spin_sliced(operands[i], inputs[i], spaces[i], spin, nmo) for i in range(len(operands))]
        expansion.append((sign, args, spin))
    return expansion


def orbital_free_letters(subscripts, spaces):
    head, _, output = subscripts.partition('->')
    kind = {}
    for letters, space in zip(head.split(','), spaces.split(',')):
        kind.update(zip(letters, space))
    return [letter for letter in output if kind.get(letter) == 'o']


def keepspin_blocks(subscripts, spaces, operands, nmo, contract):
    free = orbital_free_letters(subscripts, spaces)
    result = {}
    for sign, args, spin in keepspin_einsum(subscripts, spaces, operands, nmo):
        key = tuple(spin[letter] for letter in free)
        value = contract(subscripts, *args)
        if sign < 0:
            value = -value
        result[key] = value if key not in result else result[key] + value
    return result


def keepspin_assemble(subscripts, spaces, result):
    free = orbital_free_letters(subscripts, spaces)
    if not free:
        value, = result.values()
        return value
    output = subscripts.partition('->')[2]
    axes = [output.index(letter) for letter in free]
    for depth in reversed(range(len(free))):
        grouped = {}
        for key, value in result.items():
            grouped.setdefault(key[:depth], {})[key[depth]] = value
        result = {}
        for prefix, halves in grouped.items():
            if len(halves) != 2:
                raise ValueError('a spin block is missing from a keepspin result')
            result[prefix] = concatenate([halves[0], halves[1]], axis=axes[depth])
    return result[()]
