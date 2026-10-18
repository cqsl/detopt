import operator
from itertools import product
import jax.numpy as jnp
from flax import struct


def _total(terms):
    return sum(terms[1:], terms[0])


@struct.dataclass
class Polynomial:
    coefficients: tuple
    max_degree: int = struct.field(pytree_node=False)
    __array_priority__ = 1000

    @classmethod
    def constant(cls, value, max_degree):
        return cls((value,), max_degree)

    @property
    def dtype(self):
        return self.coefficients[0].dtype

    @property
    def shape(self):
        return self.coefficients[0].shape

    @property
    def T(self):
        return self.map(lambda value: value.T)

    def map(self, function):
        return Polynomial(tuple(function(value) for value in self.coefficients), self.max_degree)

    def astype(self, dtype):
        return self.map(lambda value: value.astype(dtype))

    def conj(self):
        return self.map(lambda value: value.conj())

    def transpose(self, *axes):
        return self.map(lambda value: value.transpose(*axes))

    def __getitem__(self, index):
        return self.map(lambda value: value[index])

    def __neg__(self):
        return self.map(operator.neg)

    def __add__(self, other):
        return _add(self, other)

    def __radd__(self, other):
        return _add(other, self)

    def __sub__(self, other):
        return _add(self, -_as_polynomial(other, self.max_degree))

    def __rsub__(self, other):
        return _add(other, -self)

    def __mul__(self, other):
        return _convolve(self, other, operator.mul)

    def __rmul__(self, other):
        return _convolve(other, self, operator.mul)

    def __matmul__(self, other):
        return _convolve(self, other, operator.matmul)

    def __rmatmul__(self, other):
        return _convolve(other, self, operator.matmul)


def _as_polynomial(value, max_degree):
    if isinstance(value, Polynomial):
        if value.max_degree != max_degree:
            raise ValueError('polynomial degree limits must match')
        return value
    return Polynomial.constant(value, max_degree)


def _common_degree(*operands):
    degrees = {operand.max_degree for operand in operands if isinstance(operand, Polynomial)}
    if len(degrees) != 1:
        raise ValueError('an operation requires at least one polynomial and matching degree limits')
    return degrees.pop()


def _add(left, right):
    max_degree = _common_degree(left, right)
    left, right = (_as_polynomial(left, max_degree), _as_polynomial(right, max_degree))
    length = max(len(left.coefficients), len(right.coefficients))
    coefficients = tuple((left.coefficients[degree] if degree < len(left.coefficients) else 0) + (right.coefficients[degree] if degree < len(right.coefficients) else 0) for degree in range(length))
    return Polynomial(coefficients, max_degree)


def _convolve(left, right, operation):
    max_degree = _common_degree(left, right)
    left, right = (_as_polynomial(left, max_degree), _as_polynomial(right, max_degree))
    length = min(len(left.coefficients) + len(right.coefficients) - 1, max_degree + 1)
    coefficients = []
    for degree in range(length):
        terms = [operation(left.coefficients[i], right.coefficients[degree - i]) for i in range(len(left.coefficients)) if 0 <= degree - i < len(right.coefficients)]
        coefficients.append(_total(terms))
    return Polynomial(tuple(coefficients), max_degree)


def _coefficient_or_zero(polynomial, degree):
    return polynomial.coefficients[degree] if degree < len(polynomial.coefficients) else jnp.zeros_like(polynomial.coefficients[0])


def _nested_polynomials(value):
    if isinstance(value, Polynomial):
        return [value]
    if isinstance(value, (tuple, list)):
        return [polynomial for part in value for polynomial in _nested_polynomials(part)]
    return []


def _nested_coefficient(value, degree):
    if isinstance(value, Polynomial):
        return _coefficient_or_zero(value, degree)
    if isinstance(value, tuple):
        return tuple(_nested_coefficient(part, degree) for part in value)
    if isinstance(value, list):
        return [_nested_coefficient(part, degree) for part in value]
    return value if degree == 0 else jnp.zeros_like(value)


def block(arrays):
    polynomials = _nested_polynomials(arrays)
    if not polynomials:
        return jnp.block(arrays)
    max_degree = _common_degree(*polynomials)
    length = min(max(len(polynomial.coefficients) for polynomial in polynomials), max_degree + 1)
    return Polynomial(tuple(jnp.block(_nested_coefficient(arrays, degree)) for degree in range(length)), max_degree)


def concatenate(arrays, axis=0):
    polynomials = [value for value in arrays if isinstance(value, Polynomial)]
    if not polynomials:
        return jnp.concatenate(arrays, axis=axis)
    max_degree = _common_degree(*polynomials)
    values = [_as_polynomial(value, max_degree) for value in arrays]
    length = min(max(len(value.coefficients) for value in values), max_degree + 1)
    return Polynomial(tuple(jnp.concatenate([_coefficient_or_zero(value, degree) for value in values], axis=axis) for degree in range(length)), max_degree)


def constant_like(value, reference):
    return Polynomial.constant(value, reference.max_degree) if isinstance(reference, Polynomial) else value

_SPARE_INDICES = 'STUVWXYZ'


def _einsum(subscripts, *operands, optimize='optimal'):
    if len(operands) != 2 or '->' not in subscripts:
        return jnp.einsum(subscripts, *operands, optimize=optimize)
    real, complex_operand = operands
    if jnp.iscomplexobj(real) or not jnp.iscomplexobj(complex_operand):
        return jnp.einsum(subscripts, *operands, optimize=optimize)
    if jnp.size(real) <= jnp.size(complex_operand):
        return jnp.einsum(subscripts, *operands, optimize=optimize)
    inputs, output = subscripts.split('->')
    left, right = inputs.split(',')
    axis = next((letter for letter in _SPARE_INDICES if letter not in subscripts), None)
    if axis is None:
        return jnp.einsum(subscripts, *operands, optimize=optimize)
    stacked = jnp.stack([complex_operand.real, complex_operand.imag])
    parts = jnp.einsum(f'{left},{axis}{right}->{axis}{output}', real, stacked, optimize=optimize)
    return parts[0] + 1j * parts[1]


def contract(subscripts, *operands, optimize='optimal'):
    polynomials = [operand for operand in operands if isinstance(operand, Polynomial)]
    if not polynomials:
        return _einsum(subscripts, *operands, optimize=optimize)
    max_degree = _common_degree(*polynomials)
    positions = [index for index, operand in enumerate(operands) if isinstance(operand, Polynomial)]
    lengths = [len(operands[index].coefficients) for index in positions]
    length = min(sum(size - 1 for size in lengths), max_degree) + 1
    terms = [[] for _ in range(length)]
    for degrees in product(*(range(size) for size in lengths)):
        degree = sum(degrees)
        if degree >= length:
            continue
        arguments = list(operands)
        for position, coefficient in zip(positions, degrees):
            arguments[position] = operands[position].coefficients[coefficient]
        terms[degree].append(_einsum(subscripts, *arguments, optimize=optimize))
    return Polynomial(tuple(_total(values) for values in terms), max_degree)


def outer(left, right):
    if isinstance(left, Polynomial) or isinstance(right, Polynomial):
        return _convolve(left, right, jnp.outer)
    return jnp.outer(left, right)


def result_type(*operands):
    values = [coefficient for operand in operands for coefficient in (operand.coefficients if isinstance(operand, Polynomial) else (operand,))]
    return jnp.result_type(*values)


def coeff_at(value, null):
    if not isinstance(value, Polynomial):
        return value
    if null is None:
        return value.coefficients[0]
    stacked = jnp.stack(value.coefficients)
    index = jnp.minimum(null, len(value.coefficients) - 1)
    return jnp.where(null < len(value.coefficients), stacked[index], jnp.zeros_like(stacked[0]))


def trace(operand):
    return operand.map(jnp.trace) if isinstance(operand, Polynomial) else jnp.trace(operand)


def zeros_like(operand):
    return operand.map(jnp.zeros_like) if isinstance(operand, Polynomial) else jnp.zeros_like(operand)
