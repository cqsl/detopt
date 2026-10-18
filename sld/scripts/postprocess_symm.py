from sld.scripts._env import setup_jax

setup_jax()

import jax
import jax.numpy as jnp

import argparse
import numpy as np

from sld.utils.io import record_settings, save_h5
from sld.scripts.postprocess import add_common_args, load
from sld.scripts._helpers import add_backend_args, compute_postprocess_mels, POSTPROCESS_SETTINGS
from sld.scripts._symm_proj import add_projection_args, projection_grid, projection_setup
from sld.scripts.show_symm import compute_data

from sld.symmetry_projection.kernels import get_mo_trafo
from sld.symmetry_projection.projection_setup import prep_pg_symm, table_to_h5


def est_mo_symm(trafo, impls, X, mo_coeff, mo_coeff_symm, args, label=None):
    # trafo: orbital transformation
    # mo_coeff: basis of the determinants
    # mo_coeff_symm: basis of the transformation
    trafo = get_mo_trafo(trafo, mo_coeff, mo_coeff_symm)
    PX = X.transform(trafo, trafo)
    pp, _ = compute_postprocess_mels(impls, X, {}, args, Y=PX, label=label)
    return pp


def compute_mels_post(impls, hamiltonian, X, mol, args, nelec_expected, nao, arrays_angmom_mo, arrays_dip_mo, mo_coeff, xyzstr, precomputed=None):
    data = {}
    precomputed = dict(precomputed or {})
    postprocess_data = {}
    print('Symmetry group used:', mol.groupname)

    def reuse(*keys):
        if not all(k in precomputed for k in keys):
            return False
        for k in keys:
            data[k] = precomputed[k]
        print("using precomputed mels for", ", ".join(keys))
        return True

    if args.skipvar:
        if not reuse('o', 'e'):
            print("computing energy")
            data['o'], data['e'] = compute_postprocess_mels(impls, X, hamiltonian, args, label="H")
        data['e2'] = precomputed['e2'] if 'e2' in precomputed else np.full_like(data['e'], fill_value=jnp.inf)
    else:
        if 'e2' in precomputed and not np.all(np.isfinite(precomputed['e2'])):
            del precomputed['e2']
        if not reuse('o', 'e2', 'e'):
            print("computing energy and variance")
            data['o'], data['e2'], data['e'] = compute_postprocess_mels(impls, X, hamiltonian, args, squared=True, label="H^2")

    if not reuse('s2'):
        print("computing S^2")
        N_u, N_d = nelec_expected
        arrays_s2 = {'Enuc': 0.5 * (N_u + N_d) + 0.25 * (N_u - N_d) * (N_u - N_d), 'w': -jnp.einsum("ij,ab->iabj", jnp.eye(nao), jnp.eye(nao))}
        _, data['s2'] = compute_postprocess_mels(impls, X, arrays_s2, args, label="S^2")

    ops_dict, table2, rot_mat, is_linear = prep_pg_symm(xyzstr, mol.basis, cart=mol.cart)
    postprocess_data.update(table_to_h5(mol.groupname, ops_dict, table2, is_linear))
    if is_linear and not reuse('lz'):
        arrays_angmom_paxis = jax.tree.map(lambda *x: np.einsum('a...,a->...', np.array(x), rot_mat[-1]), *arrays_angmom_mo)
        print("computing ", 'lz')
        _, data['lz'] = compute_postprocess_mels(impls, X, arrays_angmom_paxis, args, label="lz")

    for k, ao_trafo in ops_dict.items():
        if reuse(k):
            continue
        print("computing ", k)
        ao_coeff = np.eye(mo_coeff.shape[0])
        data[k] = est_mo_symm(ao_trafo, impls, X, mo_coeff, ao_coeff, args, label=k)

    # in the original frame; we could rotate to bring principal axis to z...
    if not args.skipdip:
        for i, k in enumerate(['mux', 'muy', 'muz']):
            if reuse(k):
                continue
            print("computing ", k)
            _, data[k] = compute_postprocess_mels(impls, X, arrays_dip_mo[i], args, label=k)

    return data, postprocess_data, ops_dict, is_linear

def save_state_matrix(postprocess_data, key, matrix, overlap, bases, save_mels=False):
    # like compute_data, for a single operator
    if save_mels:
        postprocess_data[key + "_mels"] = matrix
    for c, suffix in bases:
        norm = np.einsum("ai,ij,aj->a", c.conj(), overlap, c, optimize=True)
        reduced = (c.conj() @ matrix) @ c.T
        postprocess_data[key + "_matrix" + suffix] = reduced
        postprocess_data[key + suffix] = np.diag(reduced) / norm


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_backend_args(parser)
    parser.add_argument("--skipdip", action="store_true", help="skip computation of dipole moments")
    add_projection_args(parser)

    args = parser.parse_args()

    for fname in args.out:
        print()
        print(fname)

        impls, X, hamiltonian, nelec_expected, nao, (X_orig, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data, cart, arrays_dip_mo, arrays_angmom_mo, mol, _postprocess_data_old, nfreeze) = load(fname, args, symmetry=True)
        if not isinstance(X.nelec, tuple):
            raise NotImplementedError("S^2 for GHF")

        X = X.normalize_dets()
        c = X.coeffs

        # reuse the determinant matrix elements of an earlier run, everything else is recomputed
        precomputed = {k[:-len('_mels')]: v for k, v in _postprocess_data_old.items() if k.endswith('_mels')}

        data, postprocess_data, ops_dict, is_linear = compute_mels_post(
            impls, hamiltonian, X, mol, args, nelec_expected, nao, arrays_angmom_mo,
            arrays_dip_mo, mo_coeff, xyzstr, precomputed=precomputed,
        )
        # flatten
        ndet = len(c)
        data = jax.tree.map(lambda x: x.reshape(ndet, ndet), data)

        if args.save_mels:
            # save matrix elements for further postprocessing
            for k, v in data.items():
                postprocess_data[k+"_mels"] = np.array(v)


        _, postprocess_data = compute_data(data, c[None], postprocess_data, key='')
        bases = [(np.asarray(c)[None], '')]

        if args.project_spin is not None:
            trafos_lz, jobs = projection_setup(ops_dict, is_linear, mol, mo_coeff, args.project_spin)
            # a spin projection needs the GHF Hamiltonian
            hamiltonian_ghf = load(fname, args, symmetry=True, ghf_hamiltonian=True)[2] if (args.projected_energy or args.projected_variance) else None
            grid = projection_grid(
                X, args, jobs, xyzstr=xyzstr, basis=basis, cart=cart, mol=mol, mo_coeff=mo_coeff,
                ops_dict=ops_dict, trafos_lz=trafos_lz, hamiltonian=hamiltonian, hamiltonian_ghf=hamiltonian_ghf,
            )
            for key, overlap, energy, squared in grid:
                for prefix, matrix in (('', overlap), ('hp_', energy), ('h2p_', squared)):
                    if matrix is not None:
                        save_state_matrix(postprocess_data, prefix + key, np.asarray(matrix).reshape(ndet, ndet), data['o'], bases, args.save_mels)

        fname_out = fname.replace(".h5", "_post.h5")
        record_settings(postprocess_data, args, POSTPROCESS_SETTINGS)
        save_h5(fname_out, X_orig, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data=extra_data, postprocess_data=postprocess_data, cart=cart, nfreeze=nfreeze)
        print(f"saved {fname_out}, run sld-show-symm on it for the states and irreps")
if __name__ == "__main__":
    main()
