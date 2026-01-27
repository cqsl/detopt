import jax

jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, set_mesh, AxisType, reshard

mesh = Mesh(jax.devices(), "i", AxisType.Explicit)
set_mesh(mesh)

import jax.numpy as jnp

import argparse

import numpy as np

from tqdm.auto import tqdm

from sld.utils.hamiltonian import orbitals_pyscf, prepare_hamiltonian, ao2mo
from sld.utils.io import save_h5, load_h5
from sld.mels import compute_mels, get_impls

from functools import partial

from sld.slater import USSD, GSSD


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0, None))
def v_eval(f, Xi, Yj, arrays):
    if isinstance(Xi.nelec, tuple):
        return f(Xi.Xu.conj(), Xi.Xd.conj(), Yj.Xu, Yj.Xd, hamiltonian=arrays)
    else:
        return f(Xi.X.conj(), Yj.X, hamiltonian=arrays)


def compute_in_blocks(f, X, Y=None, n_results=None, batch_size=None, arrays=None, hermitian=False):
    if batch_size is None:
        return compute_mels(f, X, Y, arrays=arrays, batch_size=None)
    assert n_results is not None
    ndetx = X.ndet
    if Y is not None:
        hermitian = False
    else:
        Y = X
    ndety = Y.ndet
    dtype = X.dtype

    res = tuple(np.zeros((ndetx, ndety), dtype=dtype) for _ in range(n_results))
    if hermitian:
        ijs = np.array(list(zip(*np.triu_indices(ndetx))))
    else:
        ijs = np.array(list(np.ndindex(ndetx, ndety)))

    n = len(ijs)
    # if batch_size is None:
    #     batch_size = n
    #     m = 1
    #     rest = 0
    # else:
    #     m, rest = divmod(n, batch_size)
    m, rest = divmod(n, batch_size)

    if rest > 0:
        ijs_main = ijs[:-rest]
        ijs_rest = ijs[-rest:]
    else:
        ijs_main = ijs
        ijs_rest = None

    for ij in tqdm(ijs_main.reshape(m, batch_size, 2)):
        i, j = ij.T
        Xi = X[i]
        Yj = Y[j]
        Xi = reshard(Xi, jax.P("i"))
        Yj = reshard(Yj, jax.P("i"))
        vals = v_eval(f, Xi, Yj, arrays)
        vals = reshard(vals, jax.P())
        for k in range(n_results):
            res[k][i, j] = np.array(vals[k][:, 0, 0])
            if hermitian:
                res[k][j, i] = np.array(vals[k][:, 0, 0].T.conj())

    if ijs_rest is not None:
        i, j = ijs_rest.T
        Xi = X[i]
        Yj = Y[j]
        vals = v_eval(f, Xi, Yj, arrays)
        for k in range(n_results):
            res[k][i, j] = np.array(vals[k][:, 0, 0])
            if hermitian:
                res[k][j, i] = np.array(vals[k][:, 0, 0].T.conj())
    return tuple(map(jnp.array, res))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=str, nargs="+", help=".h5 files containing the simulation output (supports one or more different simulations)")
    parser.add_argument("--batch_size", type=int, required=False, help="how many mels to compute in parallel (max); set this if you get OOM errors")
    parser.add_argument("--verify", action="store_true", help="re-compute existing postprocessed energies and verify them for correctness")
    parser.add_argument("--load_integrals_mo", nargs=3, required=False, help="npy files for prec-computed enuc, h1e_mo, h2e_mo (optional, the default is to compute the integrals with pyscf)")
    parser.add_argument("--skipvar", action="store_true", help="skip compuattion of ⟨H²⟩")
    parser.add_argument("--nonorthogonal", action="store_true", help="use faster non-svd code; requires non-orthogonal determinants")

    parser.add_argument("--save_mels", action="store_true", help="store matrix elements in output file for future postprocessing")

    # this always uses the svd version (for now)

    args = parser.parse_args()

    for fname in args.out:
        print()
        print(fname)

        Xs_start, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data, _postprocess_data_old, cart, _ = load_h5(fname)

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
            nelec = X.nelec # TODO particle-hole
        else:
            if np.isnan(mo_coeff).any():
                raise ValueError('mo_coeff not found, please load integrals manually with --load_integrals_mo')
            nao, nelec, _, enuc, h1e_ao, h2e_ao, dip_ao, angmom_ao, mol = orbitals_pyscf(xyzstr, basis, spin, mo_coeff=mo_coeff, cart=cart, symmetry=True)
            h1e_mo = ao2mo(h1e_ao, mo_coeff)
            h2e_mo = ao2mo(h2e_ao, mo_coeff)
            del h1e_ao
            del h2e_ao

        nelec_expected = X.nelec
        particle_hole = nelec != nelec_expected

        hamiltonian = prepare_hamiltonian(enuc, h1e_mo, h2e_mo, ghf, particle_hole=particle_hole)

        if "energy_vals" in extra_data.keys():
            if extra_data["energy_vals"][:].nbytes > 1000000:
                print("removing energy vals from the dataset to save space")
                del extra_data["energy_vals"]

        impls = get_impls(ghf, args.nonorthogonal, False)
        expval = impls.expval
        expval_sq = impls.expval_sq

        X = X.normalize_dets()
        nn = X.coeffs
        postprocess_data = {}
        if True:
            if args.skipvar:
                print("computing energy")
                oo, ee = compute_in_blocks(expval, X, n_results=2, batch_size=args.batch_size, arrays=hamiltonian, hermitian=True)
                vv = np.full_like(ee, fill_value=jnp.inf)
            else:
                print("computing energy and variance")
                oo, vv, ee = compute_in_blocks(expval_sq, X, n_results=3, batch_size=args.batch_size, arrays=hamiltonian, hermitian=True)

            print("computing S^2")
            N_u, N_d = nelec_expected
            arrays_s2 = None, None, 0.5 * (N_u + N_d) + 0.25 * (N_u - N_d) * (N_u - N_d), -jnp.einsum("ij,ab->iabj", jnp.eye(nao), jnp.eye(nao))
            _, ss = compute_in_blocks(expval, X, n_results=2, arrays=arrays_s2, batch_size=args.batch_size, hermitian=True)

            ndet = len(nn)
            oo, ee, vv, ss = jax.tree.map(lambda x: x.reshape(ndet, ndet), (oo, ee, vv, ss))

            if args.save_mels:
                # save matrix elements for further postprocessing
                postprocess_data["oo_mels"] = np.array(oo)
                postprocess_data["ee_mels"] = np.array(ee)
                postprocess_data["vv_mels"] = np.array(vv)
                postprocess_data["ss_mels"] = np.array(ss)

        if True:
            denom = nn.T.conj() @ oo @ nn
            energy = nn.T.conj() @ ee @ nn / denom
            energy_sq = nn.T.conj() @ vv @ nn / denom
            s2 = nn.T.conj() @ ss @ nn / denom
            variance = energy_sq - energy**2

            print()
            print("eigval: ", eigvals[-1])
            if "s2_vals" in extra_data.keys():
                print("s2: ", extra_data["s2_vals"][-1].real)
                if "s2_loss" in extra_data:
                    print("Energy: ", eigvals[-1] - extra_data["s2_loss"] * extra_data["s2_vals"][-1].real)
            print()
            print("<H>   : ", energy.real)
            print("<H^2> : ", energy_sq.real)
            print("var   : ", variance.real)

            s = np.sqrt(s2.real + 0.25) - 0.5
            print(f"<S2>  : {s2.real:0.6f}")
            print(f"S     : {s:0.6f}")
            postprocess_data = {"s2": s2, "e": energy, "var": variance}

        if args.verify:
            for k in postprocess_data.keys():
                print(k)
                if k == 'var' and args.skipvar:
                    continue
                try:
                    np.testing.assert_allclose(_postprocess_data_old[k], postprocess_data[k], atol=1e-13)
                    print("ok", _postprocess_data_old[k] - postprocess_data[k])
                except AssertionError as e:
                    print(e)
            continue

        fname_out = fname.replace(".h5", "_var.h5")
        save_h5(fname_out, X_orig, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data=extra_data, postprocess_data=postprocess_data, cart=cart)


if __name__ == "__main__":
    main()
