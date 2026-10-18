import argparse
import re

import numpy as np

from sld.utils.io import load_h5
from sld.symmetry_projection.projection_setup import guess_states_proj, symmetry_table
from sld.symmetry_projection.fft import linear_grid, resolve_irreps


def guess_states_lookuptable(data, symbols, table, tol = 0.1, key='_full_degenerate_diag'):
    def _int(x):
        try:
            return int(x)
        except ValueError: # NAN
            return 99999999
    values = np.array([data[k+key] for k in symbols]).T
    res = []
    for val in values:
        i = tuple(_int(np.round(v.real)) for v in val)
        if len(i)==1:
            i, = i
        if  np.abs(np.array(val) - np.array(i)).max() > tol:
            s = '?'
        else:
            s = table.get(i, '?')
        res.append(s)
    return res

label_table = {'s2': 'S^2', 'mux': 'μx', 'muy': 'μy', 'muz': 'μz', 'sigma_v': 'σv', 'sigma_h': 'σh', 'sigma_d': 'σd', 'sigma_v_prime': "σv'", 'i' : 'î', 'c2':"C2", 'c2p':"C2'", 'c2pp':"C2''", 'sigma_d_yz': 'σd_yz', 'sigma_d_xz': 'σd_xz', 'sigma_d_xy': 'σd_xy'}


def guess_states(data, nstates, tables2, key='_full_degenerate_diag', tol=0.1):
    detected_irreps = []
    symbols = []
    for s2, t2 in tables2:
        if isinstance(t2, dict): # lookup table
            di = guess_states_lookuptable(data, s2, t2, tol=tol, key=key)
        else:
            di, _, _, _ = guess_states_proj(data, t2, tol=tol, key=key)
        detected_irreps.append(di)
        symbols += list(s2)
    return [''.join(di) for di in list(zip(*detected_irreps))], symbols

def print_states(nstates, data, table2, tol=0.1, columns=None):
    # columns: extra {name: one string per state} at the end
    symmetry_columns = []
    for t in table2:
        for k in t[0]:
            label = str(label_table.get(k, k))
            symmetry_columns.append((label, max(6, len(label))))
    # the frozen-core variance for runs with frozen core orbitals
    e2 = 'e2_fc' if 'e2_fc' in data else 'e2'
    # Sz only when it differs between the states (GHF)
    sz = 'sz' in data and np.ptp(np.real(data['sz'])) > 1e-6
    header = f'k  | {"E [a.u.]":>14} | {("var_fc" if e2 == "e2_fc" else "var") + " [a.u.]":>13} ' + (f'| {"Sz":>6} ' if sz else '')
    for label, width in symmetry_columns:
        header += f'| {label:>{width}} '
    header += f'| {"State":6} | {"E-E_00 [eV]":>10}'
    columns = columns or {}
    widths = {name: max(len(name), *map(len, values)) for name, values in columns.items()}
    header += "".join(f" | {name:<{widths[name]}}" for name in columns)
    print(header)
    state_strs, table2_keys = guess_states(data, nstates, table2, key='', tol=tol)
    for i in range(nstates):
        values = [[data[k][i] for k in t[0]] for t in table2]
        state_str = state_strs[i]
        au2ev = 27.2114079527
        ee = float((data['e'][i] - data['e'].min()).real * au2ev)
        var = float((data[e2][i] - data['e'][i]**2).real) if e2 in data else np.nan
        ei = data['e'][i].real
        row = f'{i:>02} | {ei:>14.9f} | {var.real:>13.9f} ' + (f'| {float(data["sz"][i].real):>6.3f} ' if sz else '')
        column_index = 0
        for v in values:
            for x in v:
                width = symmetry_columns[column_index][1]
                row += f'| {x.real:>{width}.3f} '
                column_index += 1
        row += f'| {state_str:<6} | {ee:>11.2f}'
        row += "".join(f" | {values[i]:<{widths[name]}}" for name, values in columns.items())
        print(row)

def compute_transition_dipole_moment(data, c, mu_key='mux'):
    inv_norms = (1 / np.sqrt(np.einsum('ai,aj,ij->a', c.conj(), c, data['o'], optimize=True))).real
    return np.einsum('ai,bj,a,b,ij->ab', c.conj(), c, inv_norms , inv_norms, data[mu_key], optimize=True)


def compute_data(data, c, postprocess_data, key='_full_degenerate_diag', *, save_state_mels=True):
    denom = np.einsum("ai,ij,aj->a", c.conj(), data['o'], c, optimize=True)
    data_out = {k: np.einsum("ai,ij,aj->a", c.conj(), v, c, optimize=True) / denom for k, v in data.items()}

    postprocess_data["c"+key] = np.array(c)

    if save_state_mels:
        for name, matrix in data.items():
            matrix = np.asarray(matrix)
            if not np.isfinite(matrix).all():
                raise ValueError(f"Nonfinite matrix elements for {name}")
            postprocess_data[name+"_matrix"+key] = (c.conj() @ matrix) @ c.T

    for k, v in data_out.items():
        postprocess_data[k+key] = np.array(v)

    for k in ['mux', 'muy', 'muz']:
        if k in data.keys():
            postprocess_data['t'+k+key] = np.array(compute_transition_dipole_moment(data, c, mu_key=k))

    return data_out, postprocess_data


def state_matrices(data, suffix):
    # operator matrices in the basis of the states, saved by sld-postprocess-symm
    pattern = re.compile(r"^(?P<name>.+)_matrix" + re.escape(suffix) + "$")
    return {m["name"]: np.asarray(value) for key, value in data.items() if (m := pattern.match(key))}


def state_values(matrices, U):
    norm = np.einsum("ai,ij,aj->a", U.conj(), matrices["o"], U)
    return {k: np.einsum("ai,ij,aj->a", U.conj(), m, U) / norm for k, m in matrices.items()}


def best_weights(spins, irreps, weights):
    # the largest weight of each state over all spins and irreps, with its label, aligned
    labels, values = [], []
    for state in range(weights.shape[2]):
        i, j = np.unravel_index(np.argmax(weights[:, :, state]), weights.shape[:2])
        labels.append(("" if spins is None else str(2 * spins[i] + 1)) + irreps[j])
        values.append(weights[i, j, state])
    width = max(map(len, labels))
    return [f"{label:<{width}} {value:.3f}" for label, value in zip(labels, values)]


def print_weights(spins, irreps, weights):
    # weights: spins x irreps x states, spins is None for the point group alone (summed over all spins)
    nstates = weights.shape[2]
    multiplicities = ["all"] if spins is None else [str(2 * spin + 1) for spin in spins]
    width = max(13, max(map(len, irreps)) + 2)
    header = f"{'state':>5} {'2S+1':>5} " + "".join(f"{irrep:>{width}}" for irrep in irreps) + "  best"
    print(header)
    print("-" * len(header))
    best = best_weights(spins, irreps, weights)
    for state in range(nstates):
        for i, multiplicity in enumerate(multiplicities):
            label = str(state) if i == 0 else ""
            print(f"{label:>5} {multiplicity:>5} " + "".join(f"{value:>{width}.8f}" for value in weights[i, :, state]) + (f"  {best[state]}" if i == 0 else ""))
    header = f"{'2S+1':>5} " + "".join(f"{'state=' + str(state):>{width}}" for state in range(nstates))
    print()
    print(header)
    print("-" * len(header))
    if spins is not None:
        for i, multiplicity in enumerate(multiplicities):
            print(f"{multiplicity:>5} " + "".join(f"{value:>{width}.8f}" for value in weights[i].sum(axis=0)))
        print("-" * len(header))
    print(f"{'all':>5} " + "".join(f"{value:>{width}.8f}" for value in weights.sum(axis=(0, 1))))


def print_energies(spins, irreps, weights, hp_weights, tol=1e-6):
    # projected energies <H P>/<P> per state, spin and irrep, '-' where the weight is below tol
    nstates = weights.shape[2]
    multiplicities = ["all"] if spins is None else [str(2 * spin + 1) for spin in spins]
    width = max(16, max(map(len, irreps)) + 2)
    fmt = lambda w, h: f"{h / w:>{width}.9f}" if abs(w) > tol else f"{'-':>{width}}"
    header = f"{'state':>5} {'2S+1':>5} " + "".join(f"{irrep:>{width}}" for irrep in irreps) + f"{'all irreps':>{width}}"
    print(header)
    print("-" * len(header))
    for state in range(nstates):
        for i, multiplicity in enumerate(multiplicities):
            label = str(state) if i == 0 else ""
            w, h = weights[i, :, state], hp_weights[i, :, state]
            print(f"{label:>5} {multiplicity:>5} " + "".join(fmt(*x) for x in zip(w, h)) + fmt(w.sum(), h.sum()))


def projector_weights(values, table2, is_linear, max_lz=None):
    # spin and irrep weights of the states from the projector values PS{S}_..., None if there are none
    if is_linear:
        grid = linear_grid(values)
        return resolve_irreps(grid, max_lz)[:3] if grid else None
    symbols, (irreps, _, characters) = table2[1]
    irreps = [irrep.replace("_", "") for irrep in irreps]
    weights = lambda prefix: np.einsum("os,go->gs", np.array([values[prefix + symbol] for symbol in symbols]), characters).real
    spins = sorted({int(m[1]) for key in values if (m := re.fullmatch(r"PS(\d+)_(.+)", key)) and m[2] in symbols})
    if spins:
        return np.array(spins), irreps, np.array([weights(f"PS{spin}_") for spin in spins])
    # without --project_spin only the point group, summed over all spins
    if all(symbol in values for symbol in symbols):
        return None, irreps, weights("")[None]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=str, nargs="+", help="_post.h5 files written by sld-postprocess-symm")
    parser.add_argument("--tol", type=float, default=0.1, help="tolerance for assigning an irrep (default: 0.1)")
    parser.add_argument("--max_lz", type=int, help="largest |Lz| sector of the linear irrep weights (default: all)")
    args = parser.parse_args()

    for fname in args.out:
        print()
        print(fname)
        _, _, _, xyzstr, basis, _, _, extra_data, data, cart, _ = load_h5(fname, jax_arrays=False)
        nstates = int(extra_data.get("nstates", 1))
        table2, is_linear = symmetry_table(data, xyzstr, basis, cart=cart)
        if table2 is None:
            raise ValueError(f"no character table available for {fname}")

        # the rotated states of sld-diagonalize-symm, if any
        suffix = "" if nstates == 1 else "_full_degenerate_diag" if "c_full_degenerate_diag" in data else "_full"
        matrices = state_matrices(data, suffix)
        if "o" not in matrices or "e" not in matrices:
            raise ValueError(f"{fname} has no state matrices, run sld-postprocess-symm on the optimization output first")
        if nstates > 1:
            print("rotated states" if suffix == "_full_degenerate_diag" else "states, not rotated (see sld-diagonalize-symm)")

        values = state_values(matrices, np.eye(len(matrices["o"])))
        print_states(nstates, values, table2, tol=args.tol)
        weights = projector_weights(values, table2, is_linear, args.max_lz)
        if weights is not None:
            print("\nspin and irrep projector weights:")
            print_weights(*weights)
        # with --projected_energy the same combination of the <H P> values gives the projected energies
        hp = {key[3:]: value for key, value in values.items() if key.startswith("hp_")}
        hp_weights = projector_weights(hp, table2, is_linear, args.max_lz) if hp and weights is not None else None
        if hp_weights is not None:
            print("\nprojected energies <H P>/<P> [a.u.]:")
            print_energies(*weights, hp_weights[2])


if __name__ == "__main__":
    main()
