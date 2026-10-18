import numpy as np


def quad_points_cosbeta(deg):
    x, weights = np.polynomial.legendre.leggauss(deg)
    return np.arccos(x), weights


def quad_points_uniform(deg, l=0, r=2 * np.pi):
    xs = l + (r - l) * np.arange(deg) / deg
    return xs, np.full(deg, (r - l) / deg)
