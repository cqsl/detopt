import re

import numpy as np

# Lz grid point keys, e.g. PS0_PLz3_sigma_v
GRID_KEY = re.compile(r"^(?:PS(?P<spin>\d+)_)?PLz(?P<angle>\d+)_(?P<symbol>.+)$")
LINEAR_GROUPS = {
    "C∞v": (("E", "sigma_v"), False),
    "D∞h": (("E", "sigma_v", "i", "C2"), True),
}
LAMBDA_NAMES = ("Σ", "Π", "Δ", "Φ", "Γ", "H", "I", "K")


def _lambda_name(value):
    return LAMBDA_NAMES[value] if value < len(LAMBDA_NAMES) else f"|Lz|={value}"


def linear_grid(values):
    # {key: per-state values} -> {(spin, angle, symbol): per-state values}
    symbols = {symbol for ops, _ in LINEAR_GROUPS.values() for symbol in ops}
    grid = {}
    for key, value in values.items():
        match = GRID_KEY.match(key)
        if match is not None and match["symbol"] in symbols:
            grid[int(match["spin"] or 0), int(match["angle"]), match["symbol"]] = np.asarray(value)
    return grid


def resolve_irreps(values, max_lz=None):
    spins = sorted({spin for spin, _, _ in values})
    angles = sorted({angle for _, angle, _ in values})
    operations, has_inversion = LINEAR_GROUPS["D∞h" if any(symbol == "i" for _, _, symbol in values) else "C∞v"]
    data = np.array([[[values[spin, angle, symbol] for angle in angles] for symbol in operations] for spin in spins])
    # spins x operations x angles x states
    fourier = np.fft.fft(data, axis=2) / len(angles)
    lz = np.fft.fftfreq(len(angles), d=1.0 / len(angles)).astype(int)

    irreps, projected = [], []
    for abs_lz in range(int(np.abs(lz).max()) + 1):
        if max_lz is not None and abs_lz > max_lz:
            break
        lz_weight = (np.abs(lz) == abs_lz).astype(int)
        if abs_lz == 0:
            pg_weight = np.array([[1, 1], [1, -1]]) / 2
            labels = ("Σ+", "Σ-")
            if has_inversion:
                pg_weight = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 4
                labels = ("Σg+", "Σg-", "Σu+", "Σu-")
        elif has_inversion:
            pg_weight = np.array([[1, 0, 1, 0], [1, 0, -1, 0]]) / 2
            labels = (f"{_lambda_name(abs_lz)}g", f"{_lambda_name(abs_lz)}u")
        else:
            pg_weight = np.array([[1, 0]])
            labels = (_lambda_name(abs_lz),)
        # per-state values give real weights, state matrices the (complex) projector matrices
        result = np.einsum("son...,n,go->sg...", fourier, lz_weight, pg_weight)
        result = result.real if data.ndim == 4 else result
        irreps.extend(labels)
        projected.extend(result[:, i] for i in range(len(labels)))

    # spins x irreps x states (x states)
    weights = np.stack(projected, axis=1)
    return np.array(spins), np.array(irreps), weights, lz
