import argparse
import os

import numpy as np

from sld.utils.io import load_h5, save_h5
from sld.symmetry_projection.fft import linear_grid, resolve_irreps
from sld.symmetry_projection.projection_setup import symmetry_table
from sld.scripts.show_symm import best_weights, compute_data, print_states, print_weights, projector_weights, state_matrices, state_values

AU2EV = 27.2114079527


def point_group_operator(matrices, table2):
    # sum_n (n+1) P_n over the irreps n; hermitian even if the group operations are not, its eigenvectors are symmetry adapted
    # any distinct weights work, n+1 just makes the eigenvalue the irrep number
    symbols, (_, _, characters) = table2[1]
    projectors = [sum(chi * matrices[symbol] for chi, symbol in zip(row, symbols)) for row in characters]
    return sum((n + 1) * P for n, P in enumerate(projectors))


def operators(matrices, table2, is_linear):
    # the operators offered for diagonalization: H, S^2, pg and the group operations
    ops = {"H": matrices["e"], "S^2": matrices["s2"]}
    # Sz only for GHF, UHF states all have the same Sz
    sz, o = matrices.get("sz"), matrices["o"]
    if sz is not None and not np.allclose(sz, np.trace(sz) / np.trace(o) * o):
        ops["Sz"] = sz
    pg = not is_linear and table2 is not None and len(table2) > 1 and all(symbol in matrices for symbol in table2[1][0])
    if pg:
        ops["pg"] = point_group_operator(matrices, table2)
    # for linear molecules the projectors onto the spin and irrep sectors (Σg+, Πu, ...), from the Lz grid of --project_spin;
    # they commute like the irrep projectors of a finite group, but do not split +Λ and -Λ
    grid = linear_grid(matrices) if is_linear else None
    if grid:
        ops["pg"] = sum((n + 1) * P for n, P in enumerate(resolve_irreps(grid)[2].reshape(-1, *o.shape)))
    symbols = ["lz", "i", "sigma_v"] if is_linear else table2[1][0] if pg else []
    ops.update({symbol: matrices[symbol] for symbol in symbols if symbol in matrices and symbol != "E"})
    return ops


def short_name(name):
    # C_2''(1) -> c2pp1, S^2 -> s2, so both spellings work
    for old, new in (("_", ""), ("^", ""), ("(", ""), (")", ""), ("''", "pp"), ("'", "p")):
        name = name.replace(old, new)
    return name.lower()


def find_operator(ops, name):
    names = {short_name(key): key for key in ops}
    if short_name(name) not in names:
        raise ValueError(f"unknown operator {name}, use one of {list(ops)}")
    return names[short_name(name)]


def diagonalize_block(U, operator, states):
    # rotate the selected states so that the operator is diagonal among them
    states = list(states)
    M = U[states].conj() @ operator @ U[states].T
    _, v = np.linalg.eigh((M + M.conj().T) / 2)
    U = U.copy()
    U[states] = v.T @ U[states]
    return U


def degenerate_blocks(U, matrices, etol):
    # groups of consecutive states (in energy) closer than etol
    e = state_values(matrices, U)["e"].real
    order = np.argsort(e)
    blocks, block = [], [order[0]]
    for i, j in zip(order, order[1:]):
        if e[j] - e[i] < etol:
            block.append(j)
        else:
            blocks.append(block)
            block = [j]
    blocks.append(block)
    return [b for b in blocks if len(b) > 1]


def _step(U, ops, name, states, reason):
    # diagonalize one operator among the states and print what it did
    U = diagonalize_block(U, ops[name], states)
    values = ", ".join(f"{(U[k].conj() @ ops[name] @ U[k]).real:.2f}" for k in states)
    print(f"states {', '.join(map(str, states))}: {reason}, diagonalizing {name} -> {values}")
    return U


def expectation(U, matrices, operator):
    return (np.einsum("ai,ij,aj->a", U.conj(), operator, U) / np.einsum("ai,ij,aj->a", U.conj(), matrices["o"], U)).real


def auto(U, matrices, ops, etol=1.5e-3, stol=3e-1):
    # within near-degenerate blocks diagonalize S^2, then Sz (GHF), the point group and Lz (linear), each among the states with the same values of the ones before
    blocks = degenerate_blocks(U, matrices, etol)
    if not blocks:
        print(f"no near-degenerate states (etol = {etol * AU2EV:.4f} eV)")
    for block in blocks:
        block = sorted(block)
        e = state_values(matrices, U)["e"].real[block]
        U = _step(U, ops, "S^2", block, f"ΔE = {(e.max() - e.min()) * AU2EV:.4f} eV < etol = {etol * AU2EV:.4f} eV")
        done = ["S^2"]
        for name in ("Sz", "pg", "lz"):
            if name not in ops:
                continue
            values = [expectation(U, matrices, ops[previous]) for previous in done]
            for k in block:
                same = [l for l in block if all(abs(v[l] - v[k]) < stol for v in values)]
                if len(same) > 1 and k == min(same):
                    U = _step(U, ops, name, same, "same " + " and ".join(done))
            done.append(name)
    return U


def rotated_name(fname):
    # x_post.h5 -> x_post_rotated.h5, also for an already rotated input
    stem = fname[:-3] if fname.endswith(".h5") else fname
    stem = stem[:-len("_rotated")] if stem.endswith("_rotated") else stem
    return stem + "_rotated.h5"


class Rotation:
    def __init__(self, fname, output=None):
        self.fname = fname
        self.output = output or rotated_name(fname)
        if os.path.abspath(self.output) == os.path.abspath(fname):
            raise ValueError(f"the rotated states would overwrite the input {fname}, choose another --output")
        self.loaded = load_h5(fname, jax_arrays=False)
        _, _, _, xyzstr, basis, _, _, extra_data, data, cart, _ = self.loaded
        if int(extra_data.get("nstates", 1)) < 2:
            raise ValueError(f"{fname} has a single state, nothing to rotate")
        self.matrices = state_matrices(data, "_full")
        if "o" not in self.matrices or "e" not in self.matrices:
            raise ValueError(f"{fname} has no state matrices, run sld-postprocess-symm on the optimization output first")
        self.table2, self.is_linear = symmetry_table(data, xyzstr, basis, cart=cart)
        self.operators = operators(self.matrices, self.table2, self.is_linear)
        k = len(self.matrices["o"])
        # continue from a saved rotation
        self.U = np.asarray(data.get("rotation_full_degenerate_diag", np.eye(k)), dtype=complex)
        self.history = []
        self.saved = False

    def values(self):
        return state_values(self.matrices, self.U)

    def apply(self, U):
        self.history.append(self.U)
        self.U = U
        self.saved = False

    def diagonalize(self, states, operator):
        self.apply(diagonalize_block(self.U, self.operators[find_operator(self.operators, operator)], states))

    def auto(self, **kwargs):
        self.apply(auto(self.U, self.matrices, self.operators, **kwargs))

    def energy_changes(self):
        # energy change of each state against the unrotated state it overlaps most with, so the order does not matter
        original = np.diag(self.matrices["e"]).real
        origin = np.argmax(np.abs(self.U) ** 2, axis=1)
        shifts = [f"{(e - original[l]) * AU2EV:+.3f}" for e, l in zip(self.values()["e"].real, origin)]
        width = max(map(len, shifts))
        return [f"{shift:>{width}}" for shift in shifts]

    def sort(self):
        self.apply(self.U[np.argsort(self.values()["e"].real)])

    def reset(self):
        self.apply(np.eye(len(self.U), dtype=complex))

    def undo(self):
        if self.history:
            self.U = self.history.pop()

    def save(self):
        Xs, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data, data, cart, nfreeze = self.loaded
        _, data = compute_data(self.matrices, self.U, data, key="_full_degenerate_diag")
        data["c_full_degenerate_diag"] = self.U @ data["c_full"]
        data["rotation_full_degenerate_diag"] = self.U
        if "trdm_full" in data:
            data["trdm_full_degenerate_diag"] = np.einsum("ac,scdpq,bd->sabpq", self.U.conj(), data["trdm_full"], self.U, optimize=True)
        save_h5(self.output, Xs, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data=extra_data, postprocess_data=data, cart=cart, nfreeze=nfreeze)
        self.saved = True
        print(f"saved the rotated states to {self.output}")


def shell(rotation, tol=0.1):
    def table():
        values = rotation.values()
        w = projector_weights(values, rotation.table2, rotation.is_linear)
        columns = {} if w is None else {"best": best_weights(*w)}
        columns["ΔE [eV]"] = rotation.energy_changes()
        print_states(len(rotation.U), values, rotation.table2, tol=tol, columns=columns)

    def weights():
        w = projector_weights(rotation.values(), rotation.table2, rotation.is_linear)
        if w is None:
            print("no projector weights, for linear molecules run sld-postprocess-symm with --project_spin")
        else:
            print_weights(*w)

    def diagonalize(states, operator="H"):
        rotation.diagonalize(states, operator)
        table()

    def auto(etol=1.5e-3, stol=3e-1):
        rotation.auto(etol=etol, stol=stol)
        table()

    def sort():
        rotation.sort()
        table()

    def undo():
        rotation.undo()
        table()

    def reset():
        rotation.reset()
        table()

    save = rotation.save

    names = [short_name(op) for op in rotation.operators]
    spatial = "lz" if "lz" in names else "pg" if "pg" in names else names[-1]
    print(f"""
commands:
  table()                        print the states
  weights()                      print the spin and irrep projector weights (only the point group without --project_spin)
  diagonalize(states, operator)  diagonalize an operator among the states (the k of the table)
  auto()                         diagonalize s2, then {", then ".join(n for n in ("sz", "pg", "lz") if n in names)}, among near-degenerate states
  sort()                         order the states by energy
  undo(), reset()                undo the last step, go back to the unrotated states
  save()                         write the rotated states to {rotation.output}

operators: {", ".join(names)}

for example:
  diagonalize([2, 3], 's2')
  diagonalize([2, 3], '{spatial}')
  diagonalize([4, 5], '{names[-1]}')
  save()
""")
    table()
    from IPython import embed
    embed()
    if rotation.history and not rotation.saved and input(f"save the rotated states to {rotation.output}? [y/N] ").strip().lower() == "y":
        rotation.save()


def main():
    parser = argparse.ArgumentParser(description="diagonalize S^2, Lz or the point group among near-degenerate excited states, using the state matrices saved by sld-postprocess-symm")
    parser.add_argument("out", type=str, help="_post.h5 file written by sld-postprocess-symm")
    parser.add_argument("--auto", action="store_true", help="diagonalize S^2 and then Lz (linear) or the point group in near-degenerate blocks and save, without the shell")
    parser.add_argument("--etol", type=float, default=1.5e-3, help="energy difference in a.u. below which states count as degenerate for --auto (default: 1.5e-3)")
    parser.add_argument("--output", type=str, help="file for the rotated states (default: the input name with _rotated.h5), the input is never changed")
    parser.add_argument("--tol", type=float, default=0.1, help="tolerance for assigning an irrep in the printed table (default: 0.1)")
    args = parser.parse_args()

    rotation = Rotation(args.out, args.output)
    if args.auto:
        print_states(len(rotation.U), rotation.values(), rotation.table2, tol=args.tol)
        print()
        rotation.auto(etol=args.etol)
        print()
        print_states(len(rotation.U), rotation.values(), rotation.table2, tol=args.tol)
        rotation.save()
    else:
        shell(rotation, tol=args.tol)


if __name__ == "__main__":
    main()
