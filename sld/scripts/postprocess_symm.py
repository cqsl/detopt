from sld.scripts._env import setup_jax

setup_jax()

import jax
import jax.numpy as jnp

import argparse
import gc
import numpy as np

from sld.utils.io import record_settings, save_h5
from sld.solvers import generalized_eigh_lowrank
from sld.scripts.postprocess import add_common_args, frozen_core, load
from sld.scripts._helpers import add_backend_args, compute_postprocess_mels, s2_arrays, positive_int, progress_wanted, POSTPROCESS_SETTINGS
from sld.scripts._symm_proj import add_projection_args, projection_grid, projection_setup
from sld.scripts.show_symm import compute_data

from sld.mels._src.pairwise import unstable as unstable_
from sld.mels._src.pairwise.stable.common import A_inv_det_stable_1
from tqdm.auto import tqdm
from sld.symmetry_projection.kernels import get_mo_trafo
from sld.symmetry_projection.projection_setup import prep_pg_symm, table_to_h5


def est_mo_symm(trafo, impls, X, mo_coeff, mo_coeff_symm, args, label=None):
    # trafo: orbital transformation
    # mo_coeff: basis of the determinants
    # mo_coeff_symm: basis of the transformation
    trafo = get_mo_trafo(trafo, mo_coeff, mo_coeff_symm)
    PX = X.transform(trafo, trafo) if isinstance(X.nelec, tuple) else X.transform(jnp.kron(jnp.eye(2), trafo))
    pp, _ = compute_postprocess_mels(impls, X, {}, args, Y=PX, label=label)
    return pp


def compute_mels_post(impls, hamiltonian, X, mol, args, nelec_expected, nao, arrays_angmom_mo, arrays_dip_mo, mo_coeff, xyzstr, precomputed=None, frozen=None):
    # frozen: (X, hamiltonian) of the active orbitals, H and H^2 then use the frozen-core hamiltonian and H^2 is stored as e2_fc
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

    X_h, hamiltonian_h = (X, hamiltonian) if frozen is None else frozen
    e2 = 'e2' if frozen is None else 'e2_fc'
    # without H^2 no e2 keys are saved; older files store it as non-finite placeholders, which are dropped
    if e2 in precomputed and not np.all(np.isfinite(precomputed[e2])):
        del precomputed[e2]
    if args.skipvar:
        reuse(e2)
        if not reuse('o', 'e'):
            print("computing energy")
            data['o'], data['e'] = compute_postprocess_mels(impls, X_h, hamiltonian_h, args, label="H")
    else:
        if not reuse('o', e2, 'e'):
            print("computing energy and variance" + ("" if frozen is None else " of the frozen-core hamiltonian"))
            data['o'], data[e2], data['e'] = compute_postprocess_mels(impls, X_h, hamiltonian_h, args, squared=True, label="H^2")

    if not reuse('s2'):
        print("computing S^2")
        o, data['s2'] = compute_postprocess_mels(impls, X, s2_arrays(X, nelec_expected, nao), args, label="S^2")
        # the frozen orbitals are the same in all determinants, so the overlaps agree
        if frozen is not None and not np.allclose(np.asarray(o), np.asarray(data['o']), atol=1e-10):
            raise ValueError("the frozen-core determinants do not match the full ones")

    if not reuse('sz'):
        if isinstance(X.nelec, tuple):
            # UHF determinants are eigenstates of Sz
            data['sz'] = (X.nelec[0] - X.nelec[1]) / 2 * np.asarray(data['o'])
        else:
            print("computing Sz")
            _, data['sz'] = compute_postprocess_mels(impls, X, {'H1e': jnp.kron(jnp.diag(jnp.array([0.5, -0.5])), jnp.eye(nao))}, args, label="Sz")

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


def compute_trdm(X, bases, overlap, nonorthogonal, batch_size=None, show_progress=False):
    # transition 1-RDMs trdm[s, a, b, p, q] = <a|c^+_ps c_qs|b> between the normalized states of each basis, s = up, down
    # with M_ij = U*_i Vⱼᵀ per spin: Σ_ij c*_ai c_bj ⟨φ_i|c^+_p c_q|φ_j⟩ = Σ_i c*_ai [Σ_j c_bj Vⱼᵀ adj(M_ij) det(M̄_ij)] U*_i at [q, p],
    # so only the small adjugates are per pair and everything stays on the device
    C = jnp.asarray(np.concatenate([c for c, _ in bases]))
    Xc = X.conj()
    ku, kd = X.nelec

    def adj_det(M):
        if nonorthogonal:
            M_inv, det = unstable_.mels.inv_and_det(M)
            return M_inv * det, det
        D, A = A_inv_det_stable_1(M)
        return A * D, jnp.linalg.det(M)

    @jax.jit
    def step(T, Cc, Uu, Ud):
        def pair(uu, ud, vu, vd):
            (au, du), (ad, dd) = adj_det(uu @ vu.T), adj_det(ud @ vd.T)
            return au * dd, ad * du
        Gu, Gd = jax.vmap(jax.vmap(pair, (None, None, 0, 0)), (0, 0, None, None))(Uu, Ud, X.Xu, X.Xd)
        Wu = jnp.einsum("yj,jkn,ijkl->iynl", C, X.Xu, Gu)
        Wd = jnp.einsum("yj,jkn,ijkl->iynl", C, X.Xd, Gd)
        return T + jnp.stack([jnp.einsum("xi,iynl,ilm->xymn", Cc, Wu, Uu), jnp.einsum("xi,iynl,ilm->xymn", Cc, Wd, Ud)])

    n = X.Xu.shape[-1]
    if batch_size is None:
        batch_size = max(1, 2**25 // (X.ndet * (ku * ku + kd * kd)))
    batch_size = min(batch_size, X.ndet)
    # pad the last batch with copies of the first determinant and zero coefficients, so there is one compilation
    nbatch = -(-X.ndet // batch_size)
    idx = np.arange(nbatch * batch_size)
    Cc = jnp.where(idx < X.ndet, C.conj()[:, np.minimum(idx, X.ndet - 1)], 0)
    idx = np.where(idx < X.ndet, idx, 0)
    T = jnp.zeros((2, len(C), len(C), n, n), dtype=jnp.result_type(C, X.Xu))
    for b in tqdm(range(nbatch), desc="transition 1-RDMs", unit="batch", disable=not show_progress):
        i = idx[b * batch_size:(b + 1) * batch_size]
        T = step(T, Cc[:, b * batch_size:(b + 1) * batch_size], Xc.Xu[i], Xc.Xd[i])
    T = np.asarray(T)
    out, offset = {}, 0
    for c, suffix in bases:
        norm = np.sqrt(np.einsum("ai,ij,aj->a", c.conj(), overlap, c, optimize=True).real)
        t = T[:, offset:offset + len(c), offset:offset + len(c)]
        out["trdm" + suffix] = t / (norm[:, None] * norm[None, :])[None, :, :, None, None]
        offset += len(c)
    return out


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_backend_args(parser)
    parser.add_argument("--skipdip", action="store_true", help="skip computation of dipole moments")
    parser.add_argument("--trdm", action="store_true", help="also save the transition 1-RDMs between the states, trdm<suffix> of shape (2, k, k, n_orb, n_orb) (UHF only)")
    parser.add_argument("--batch_size_trdm", type=positive_int, help="bra determinants per step for --trdm (default: automatic)")
    add_projection_args(parser)

    args = parser.parse_args()

    for fname in args.out:
        print()
        print(fname)

        impls, X, hamiltonian, nelec_expected, nao, (X_orig, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data, cart, arrays_dip_mo, arrays_angmom_mo, mol, _postprocess_data_old, nfreeze) = load(fname, args, symmetry=True, frozen=False)
        if not isinstance(X.nelec, tuple) and args.trdm:
            raise NotImplementedError("--trdm supports UHF only")
        if args.project_spin is not None and sum(np.atleast_1d(X.nelec)) % 2:
            raise NotImplementedError("--project_spin with an odd number of electrons (half-integer S)")

        X = X.normalize_dets()
        c = X.coeffs

        # reuse the determinant matrix elements of an earlier run, everything else is recomputed
        precomputed = {k[:-len('_mels')]: v for k, v in _postprocess_data_old.items() if k.endswith('_mels')}

        # with frozen core orbitals H and H^2 use the frozen-core hamiltonian, everything else the full one
        frozen = None
        if frozen_core(X_orig, nfreeze):
            _, X_fc, hamiltonian_fc, *_ = load(fname, args, symmetry=True, frozen=True)
            frozen = (X_fc.normalize_dets(), hamiltonian_fc)
        data, postprocess_data, ops_dict, is_linear = compute_mels_post(
            impls, hamiltonian, X, mol, args, nelec_expected, nao, arrays_angmom_mo,
            arrays_dip_mo, mo_coeff, xyzstr, precomputed=precomputed, frozen=frozen,
        )
        # flatten
        ndet = len(c)
        data = jax.tree.map(lambda x: x.reshape(ndet, ndet), data)

        if args.save_mels:
            # save matrix elements for further postprocessing
            for k, v in data.items():
                postprocess_data[k+"_mels"] = np.array(v)


        if "nper" in extra_data.keys():
            print("excited states; computing individual states and full lvp")
            nper = int(extra_data["nper"])
            nstates = int(extra_data["nstates"])
            c = np.array(c.reshape(nstates, nper))
            # sum over states
            data_states = jax.tree.map(lambda x: np.einsum("ai,aibj,bj->ab", c.conj(), x.reshape(nstates, nper, nstates, nper), c, optimize=True), data)

            print("diagonalizing states")
            c_all = np.array(generalized_eigh_lowrank(data_states['e'], data_states['o'], rcond=1e-11)[1].T)

            print("diagonalizing determinants")


            c_full = np.array(generalized_eigh_lowrank(data['e'], data['o'], rcond=1e-11)[1].T[:nstates])
            gc.collect()

            _, postprocess_data = compute_data(data_states, c_all, postprocess_data, key='_all')
            _, postprocess_data = compute_data(data, c_full, postprocess_data, key='_full')
            # the states in terms of the determinants
            blocks = np.zeros((nstates, ndet), dtype=c.dtype)
            for a in range(nstates):
                blocks[a, a*nper:(a+1)*nper] = c[a]
            bases = [(c_all @ blocks, '_all'), (c_full, '_full')]
        else:
            _, postprocess_data = compute_data(data, c[None], postprocess_data, key='')
            bases = [(np.asarray(c)[None], '')]

        if args.trdm:
            print("computing the transition 1-RDMs")
            postprocess_data.update(compute_trdm(X, bases, np.asarray(data['o']), args.nonorthogonal, args.batch_size_trdm, show_progress=progress_wanted(args)))

        if args.project_spin is not None:
            # S < |Sz| has zero weight
            sz = abs(X.nelec[0] - X.nelec[1]) / 2 if isinstance(X.nelec, tuple) else 0
            if any(S < sz for S in args.project_spin):
                print(f"skipping S < |Sz| = {sz:g} in --project_spin")
            trafos_lz, jobs = projection_setup(ops_dict, is_linear, mol, mo_coeff, [S for S in args.project_spin if S >= sz])
            # a spin projection needs the GHF Hamiltonian
            hamiltonian_ghf = load(fname, args, symmetry=True, ghf_hamiltonian=True, frozen=False)[2] if (args.projected_energy or args.projected_variance) and isinstance(X.nelec, tuple) else None
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
