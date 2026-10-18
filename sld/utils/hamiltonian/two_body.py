from flax import struct


@struct.dataclass
class AntisymmetricTwoBody:

    Ha: object


def antisymmetrize_two_body(H2e):

    Ht = H2e - H2e.transpose(0, 1, 3, 2)
    Htt = Ht - Ht.transpose(1, 0, 2, 3)
    Htt = 0.5 * (Htt + Htt.transpose(2, 3, 0, 1))
    return AntisymmetricTwoBody(Htt / 4)


__all__ = ["AntisymmetricTwoBody", "antisymmetrize_two_body"]
