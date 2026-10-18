from jax.tree_util import Partial
from flax import struct


@struct.dataclass
class Impl:
    hess: Partial = None
    grad: Partial = None
    expval: Partial = None
    expval_sq: Partial = None
    hess_spin_blocks: tuple = None
