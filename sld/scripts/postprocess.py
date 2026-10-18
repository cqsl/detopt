from sld.scripts._env import setup_jax

setup_jax()

import jax
import jax.numpy as jnp

import argparse

import numpy as np

from sld.utils.hamiltonian import orbitals_pyscf, prepare_hamiltonian, ao2mo, particle_hole_trafo, freeze_core
from sld.utils.hamiltonian.orb_conversion import spinorb_one_body
from sld.utils.io import record_settings, save_h5, load_h5
from sld.mels import get_impls

from sld.slater import USSD, GSSD

from sld.scripts._helpers import add_backend_args, compute_postprocess_mels, s2_arrays, positive_int, POSTPROCESS_SETTINGS

def add_common_args(parser):
    parser.add_argument("out", type=str, nargs="+", help=".h5 files containing the simulation output (supports one or more different simulations)")
    parser.add_argument("--batch_size", type=positive_int, required=False, help="how many mels to compute in parallel (max); set this if you get OOM errors")
    parser.add_argument("--load_integrals_mo", nargs=3, required=False, help="npy files for precomputed enuc, h1e_mo, h2e_mo (optional, the default is to compute the integrals with pyscf)")
    parser.add_argument("--skipvar", action="store_true", help="skip computation of ⟨H²⟩")
    parser.add_argument("--nonorthogonal", action="store_true", help="use faster non-svd code; requires non-orthogonal determinants")
    parser.add_argument("--batch_size_var", type=positive_int, required=False, help="how many mels for <H^2> to compute in parallel (max); defaults to --batch_size; set this if you get OOM errors")
    parser.add_argument("--save_mels", action="store_true", help="also save the full determinant-space matrix elements")
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto", help="show progress per block (default: when stderr is a terminal)")
    parser.add_argument("--triangular", action=argparse.BooleanOptionalAction, default=True, help="compute only the upper triangle and mirror it (default: on)")
    parser.add_argument("--triangular_groups", type=positive_int, default=8, help="number of bra groups for the triangle (default: 8)")
    return parser

def frozen_core(Xs, nfreeze):
    # with frozen core orbitals the variance is that of the frozen-core hamiltonian (UHF only)
    return bool(nfreeze) and isinstance(Xs, tuple)


def load(fname, args, symmetry=True, impls=None, ghf_hamiltonian=False, frozen=None):
    # frozen: use the frozen-core hamiltonian on the active orbitals, default frozen_core(...); otherwise the frozen orbitals are added back
    Xs_start, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data, _postprocess_data_old, cart, nfreeze = load_h5(fname)
    frozen_core_variance = frozen_core(Xs_start, nfreeze) if frozen is None else frozen
    if frozen_core_variance and nfreeze % 2:
        raise ValueError("the frozen-core hamiltonian needs an even nfreeze")

    if isinstance(Xs_start, tuple):
        coeffs_start = jnp.ones(Xs_start[0].shape[0], dtype=Xs_start[0].dtype)
        X = USSD(*Xs_start, coeffs_start).normalize_orbs()
    else:
        coeffs_start = jnp.ones(Xs_start.shape[0], dtype=Xs_start.dtype)
        X = GSSD(Xs_start, coeffs_start).normalize_orbs()

    X_orig = Xs_start

    ghf = not isinstance(X.nelec, tuple)

    # loading pre-computed coefficients directly
    if args.load_integrals_mo is not None:
        enuc = np.load(args.load_integrals_mo[0])
        h1e_mo = np.load(args.load_integrals_mo[1])
        h2e_mo = np.load(args.load_integrals_mo[2])
        particle_hole = False
        nao = h1e_mo.shape[0]
        nelec = X.nelec
        if nfreeze is not None:
            nelec = tuple(n + nfreeze // 2 for n in X.nelec)
        dip_mo, angmom_mo, mol = [], [], None
    else:
        if np.isnan(mo_coeff).any():
            raise ValueError("mo_coeff not found, please load integrals manually with --load_integrals_mo")
        nao, nelec, _, enuc, h1e_ao, h2e_ao, dip_ao, angmom_ao, mol = orbitals_pyscf(xyzstr, basis, spin, mo_coeff=mo_coeff, cart=cart, symmetry=symmetry, charge=int(extra_data.get("charge", 0)))
        h1e_mo = ao2mo(h1e_ao, mo_coeff)
        h2e_mo = ao2mo(h2e_ao, mo_coeff)
        del h1e_ao
        del h2e_ao
        dip_mo = [ao2mo(di, mo_coeff) for di in dip_ao]
        angmom_mo = [ao2mo(ai, mo_coeff) for ai in angmom_ao]

    if nfreeze is None:
        nelec_expected = tuple(nelec) if ghf else X.nelec
        particle_hole = (sum(nelec) if ghf else nelec) != X.nelec
    else:
        # no particle hole supported when using frozen core
        particle_hole = False
        nelec_expected=nelec

    if particle_hole:
        print('using particle-hole transformation')

    if frozen_core_variance:
        ncore = nfreeze // 2
        if h1e_mo.shape != (X.Xu.shape[-1] + ncore,) * 2:
            raise ValueError("the frozen-core hamiltonian needs the full MO integrals including the frozen core")
        print("frozen core data; computing variance with the frozen-core effective Hamiltonian")
        enuc, h1e_mo, h2e_mo = freeze_core(enuc, h1e_mo, h2e_mo, nfreeze=nfreeze)
        nao = X.Xu.shape[-1]
        nelec_expected = X.nelec

    spin_orbital = ghf or ghf_hamiltonian
    hamiltonian = prepare_hamiltonian(
        enuc, h1e_mo, h2e_mo, spin_orbital, particle_hole=particle_hole,
        ghf_keepspin=spin_orbital and args.backend != "pairwise",
    )

    if "energy_vals" in extra_data.keys():
        if extra_data["energy_vals"][:].nbytes > 1000000:
            print("removing energy vals from the dataset to save space")
            del extra_data["energy_vals"]

    if impls is None:
        impls = get_impls(ghf, args.nonorthogonal)

    if particle_hole:
        arrays_dip_mo = [dict(zip(['Enuc', 'H1e'], particle_hole_trafo(0, di, None)[:2])) for di in dip_mo]
        arrays_angmom_mo = [dict(zip(['Enuc', 'H1e'], particle_hole_trafo(0, ai, None)[:2])) for ai in angmom_mo]
    elif frozen_core_variance:
        def frozen_one_body(di):
            return {'Enuc': 2 * jnp.trace(di[:ncore, :ncore]), 'H1e': di[ncore:, ncore:]}
        arrays_dip_mo = [frozen_one_body(di) for di in dip_mo]
        arrays_angmom_mo = [frozen_one_body(ai) for ai in angmom_mo]
    else:
        one_body = spinorb_one_body if ghf else (lambda x: x)
        arrays_dip_mo = [{'H1e': one_body(di)} for di in dip_mo]
        arrays_angmom_mo = [{'H1e': one_body(ai)} for ai in angmom_mo]
    del dip_mo
    del angmom_mo

    if nfreeze is not None and not frozen_core_variance:
        print("frozen core data; adding frozen orbitals back to the state")
        X = X.unfreeze(nfreeze // 2, nfreeze // 2)

    return impls, X, hamiltonian, nelec_expected, nao, (X_orig, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data, cart, arrays_dip_mo, arrays_angmom_mo, mol, _postprocess_data_old, nfreeze)


def print_standard(expvals, eigvals=[None]):
    energy = expvals['e']
    s2 = expvals['s2']
    s = np.sqrt(s2.real + 0.25) - 0.5
    print()
    print("eigval: ", eigvals[-1])
    print()
    print("<H>   : ", energy.real)
    if 'e2' in expvals:
        print("<H^2> : ", expvals['e2'].real)
        print("var   : ", (expvals['e2'] - energy**2).real)

    print(f"<S2>  : {float(s2.real):0.6f}")
    print(f"S     : {float(s.real):0.6f}")

def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_backend_args(parser)
    parser.add_argument("--verify", action="store_true", help="re-compute existing postprocessed energies and verify them for correctness")
    args = parser.parse_args()

    for fname in args.out:
        print()
        print(fname)

        impls, X, hamiltonian, nelec_expected, nao, (X_orig, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data, cart, arrays_dip_mo, arrays_angmom_mo, mol, _postprocess_data_old, nfreeze) = load(fname, args)
        fc = frozen_core(X_orig, nfreeze)
        var_key = 'var_fc' if fc else 'var'
        if args.verify and ('var' if fc else 'var_fc') in _postprocess_data_old:
            raise ValueError("frozen-core and full-Hamiltonian variances differ")

        X = X.normalize_dets()
        c = X.coeffs

        data = {}
        postprocess_data = {}
        if args.skipvar:
            print("computing energy")
            data['o'], data['e'] = compute_postprocess_mels(impls, X, hamiltonian, args, label="H")
        else:
            print("computing energy and variance")
            data['o'], data['e2'], data['e'] = compute_postprocess_mels(impls, X, hamiltonian, args, squared=True, label="H^2")

        print("computing S^2")
        _, data['s2'] = compute_postprocess_mels(impls, X, s2_arrays(X, nelec_expected, nao), args, label="S^2")

        # flatten
        ndet = len(c)
        data = jax.tree.map(lambda x: x.reshape(ndet, ndet), data)

        if args.save_mels:
            # save matrix elements for further postprocessing
            for k, v in data.items():
                postprocess_data[("e2_fc" if k == "e2" and fc else k) + "_mels"] = np.array(v)

        denom = c.T.conj() @ data['o'] @ c
        expvals = jax.tree.map(lambda x: c.T.conj() @ x @ c / denom, data)
        print_standard(expvals, eigvals=eigvals)

        postprocess_data['e'] = np.array(expvals['e'])
        if 'e2' in expvals:
            postprocess_data[var_key] = np.array(expvals['e2'] - expvals['e']**2)
        postprocess_data['s2'] = np.array(expvals['s2'])

        if args.verify:
            for k in postprocess_data.keys():
                print(k)
                try:
                    np.testing.assert_allclose(_postprocess_data_old[k], postprocess_data[k], atol=1e-13)
                    print("ok", _postprocess_data_old[k] - postprocess_data[k])
                except AssertionError as e:
                    print(e)
            continue

        fname_out = fname.replace(".h5", "_var.h5")
        record_settings(postprocess_data, args, POSTPROCESS_SETTINGS)
        save_h5(fname_out, X_orig, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data=extra_data, postprocess_data=postprocess_data, cart=cart, nfreeze=nfreeze)

if __name__ == "__main__":
    main()
