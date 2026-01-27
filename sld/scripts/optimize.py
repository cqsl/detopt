import jax

jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, set_mesh, AxisType

mesh = Mesh(jax.devices(), "i", AxisType.Explicit)
set_mesh(mesh)

import jax.numpy as jnp
import argparse
import numpy as np
from tqdm.auto import tqdm
from sld.slater import GSSD, USSD
from sld.optimize.full import optimize_step

from sld.utils.hamiltonian import orbitals_pyscf, prepare_hamiltonian
from sld.utils.hamiltonian import ao2mo
from sld.mels import get_impls

from sld.utils.io import load_h5, save_h5


def setup(inp, basis, spin, cart, start, sparse_, s2_loss, unstable, keep_ao_integrals, ghf, particle_hole, nohf, symmetry=True):
    if not unstable:
        print("WARNING: using the slow SVD code; consider using --nonorthogonal if you think it is safe to assume that the determinants will be not orthogonal")

    if sparse_:
        # in practice AO can be already quite localized, so we only support that for now
        # TODO relax and implement localized orbitals etc
        assert keep_ao_integrals  # for sparse use AO integrals for now
        assert unstable  # SVD version not implemented yet

    with open(inp, "r") as f:
        xyzstr = f.read()

    name = inp.split("/")[-1]
    extra_data = {}
    if s2_loss is not None:
        extra_data["s2_loss"] = float(s2_loss)

    if start is not None:
        Xs_start, mo_coeff, _eigvals, _xyzstr, _basis, _spin, _name, _, _, cart, _ = load_h5(start)
        if not (_basis == basis and _spin == spin and _name == name):  # and _xyzstr == xyzstr  # TODO turn it into atom and check
            raise NotImplementedError  # TODO allow different, eg for adiabatic evolution (maybe trafo of Xs to new basis via AO)
    else:
        Xs_start = None
        mo_coeff = None
    nao, nelec, mo_coeff, enuc, h1e_ao, h2e_ao, *_ = orbitals_pyscf(xyzstr, basis, spin, mo_coeff=mo_coeff, sparse_=sparse_, run_hf=not nohf, cart=cart, symmetry=symmetry)
    print(f"nelec: {nelec}; nao: {nao}")

    if particle_hole:
        assert not keep_ao_integrals  # ph trafo in ao not implemented
        nelec = nao - nelec[0], nao - nelec[1]

    if ghf:
        nelec = sum(nelec)
        no = 2 * nao
    else:
        no = nao

    # here we choose molecular orbitals for the transform to an orthognal basis
    # this choice is an arbitrary, we could e.g. also do a cholesky of the ao overlap matrix
    if not keep_ao_integrals:
        assert not sparse_  # need AO for sparse
        # transform the integrals to MO at a cost of  O(n_o^5),
        # and run simulations in MO directly
        mo_coeff_later = None
        h1e_mo = ao2mo(h1e_ao, mo_coeff)
        h2e_mo = ao2mo(h2e_ao, mo_coeff)
        del h1e_ao
        del h2e_ao
    else:
        # run simulation using AO integrals, multiply with MO later
        mo_coeff_later = mo_coeff
        h1e_mo = h1e_ao
        h2e_mo = h2e_ao

    print(f"correlating {nelec} electrons in {no} orbitals")

    print("preparing hamiltonian arrays")
    hamiltonian = prepare_hamiltonian(enuc, h1e_mo, h2e_mo, ghf, particle_hole=particle_hole, sparse_=sparse_)
    del h1e_mo
    del h2e_mo

    if s2_loss is not None:
        assert not ghf
        assert not keep_ao_integrals  # not implemented yet
        assert not sparse_  # not implemented yet

        hamiltonian = list(hamiltonian)
        # we would not need the constant term just for the loss,
        # but this way we can subtract <S^2> later to get the energy
        N_d = nelec[0]
        N_u = nelec[1]
        hamiltonian[2] = hamiltonian[2] + s2_loss * (0.5 * (N_u + N_d) + 0.25 * (N_u - N_d) * (N_u - N_d))
        I = jnp.eye(no)
        hamiltonian[3] = hamiltonian[3] - s2_loss * jnp.einsum("jk,il->ijkl", I, I)

    if Xs_start is not None:
        if isinstance(Xs_start, tuple):
            coeffs_start = jnp.ones(Xs_start[0].shape[0], dtype=Xs_start[0].dtype)
            X_start = USSD(*Xs_start, coeffs_start).normalize_orbs()
        else:
            coeffs_start = jnp.ones(Xs_start.shape[0], dtype=Xs_start.dtype)
            X_start = GSSD(Xs_start, coeffs_start).normalize_orbs()
    else:
        X_start = None

    return hamiltonian, X_start, mo_coeff, mo_coeff_later, extra_data, name, xyzstr, no, nelec


def add_common_parser_args(parser):
    parser.add_argument("--seed", type=int, required=False, help="random seed (an integer)")
    parser.add_argument("--start", type=str, required=False, help="determinant initialization (h5 file from a previous run), needs to contain <= ndet determinants")
    parser.add_argument("--prefix", type=str, default="./", help="prefix to append to the output filename (path and/or a name)")
    parser.add_argument("--real", action="store_true", help="use real parameters; if not set defaults to complex parameters")
    parser.add_argument("--batch_size", type=int, required=False, help="how many mels to compute in parallel (max); set this if you get OOM errors")
    parser.add_argument("--nonorthogonal", action="store_true", help="assume determinants are non-orthonal; uses faster code path")
    parser.add_argument("--save_interval", type=int, required=False, help="every how many steps to save the state (default: only at the end)")
    parser.add_argument("--dense_solver", action="store_true", help="use dense eigh to solve the eigenvalue problem instead of Cholesky+Lanczos")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inp", type=str, help="input .xyz file")
    parser.add_argument("basis", type=str, help="basis")
    parser.add_argument("n_steps", type=int, help="number of iterations")
    parser.add_argument("ndet", type=int, help="number of determinants")
    add_common_parser_args(parser)
    parser.add_argument("--spin", type=int, default=0, help="spin (pyscf). This is 2*M = 2*<Sz>. Defaults to 0.")
    parser.add_argument("--ghf", action="store_true", help="use generalized hartree-fock determinants; if not set the code uses UHF")
    parser.add_argument("--particle_hole", action="store_true", help="apply a particle hole transformation to the hamiltonian")
    parser.add_argument("--s2_loss", type=float, required=False, help="add a S^2 penalty term to the Hamiltonian, scaled by the value of this argument")
    parser.add_argument("--keep_ao_integrals", action="store_true", help="do not pre-transform the integrals to molecular orbitals")
    parser.add_argument("--nohf", action="store_true", help="diagonalize the overlap matrix go get orthogonal orbitals, and do NOT run a HF calculation or use MO")  # TODO better name
    parser.add_argument("--cart", action="store_true", help="use cart orbitals instead of sph")

    args = parser.parse_args()

    print(args)

    n_steps = args.n_steps
    ndet = args.ndet
    seed = args.seed
    basis = args.basis
    inp = args.inp
    prefix = args.prefix
    spin = args.spin  # 2 * Sz
    ghf = args.ghf
    start = args.start
    unstable = args.nonorthogonal
    cplx = not args.real
    batch_size = args.batch_size
    save_interval = args.save_interval
    particle_hole = args.particle_hole
    s2_loss = args.s2_loss
    keep_ao_integrals = args.keep_ao_integrals
    sparse_ = False
    nohf = args.nohf
    cart = args.cart
    cholesky = not args.dense_solver

    if cplx:
        dtype = complex
    else:
        dtype = float

    if ndet % jax.device_count() != 0:
        raise ValueError("number of determinants needs to be a multiple of the number of jax devices")

    hamiltonian, X_start, mo_coeff, mo_coeff_later, extra_data, name, xyzstr, no, nelec = setup(inp, basis, spin, cart, start, sparse_, s2_loss, unstable, keep_ao_integrals, ghf, particle_hole, nohf)

    impls = get_impls(ghf, unstable, sparse_, mo_coeff=mo_coeff_later)

    if seed is None:
        seed = np.random.randint(2**32)
        if jax.process_count() > 1:
            seed = jax.experimental.multihost_utils.broadcast_one_to_all(seed)
    key = jax.random.PRNGKey(seed)
    extra_data["seed"] = int(seed)

    key, k0 = jax.random.split(key)
    print("initializing determinants")

    if X_start is not None:
        # pad with random determinants
        X = X_start.pad(k0, ndet - len(X_start))
    elif isinstance(nelec, tuple):
        X = USSD.random(k0, ndet, nelec, no, dtype)
    else:
        X = GSSD.random(k0, ndet, nelec, no, dtype)

    eigvals = []
    postprocess_data = {}

    with tqdm(total=n_steps) as pbar:
        for step in range(1, n_steps + 1):
            key, k = jax.random.split(key)
            X, e0, info = optimize_step(
                k,
                X,
                hamiltonian,
                impls,
                batch_size=batch_size,
                cholesky=cholesky,
            )

            eigvals.append(np.array(e0).reshape(()).tolist())
            pbar.update(1)

            s2_extra = ""
            cmin = jnp.abs(X.coeffs).min().tolist()
            s2_extra = s2_extra + f" cmin:{cmin:0.1e}"
            if info is not None:
                s2_extra = s2_extra + f" iter:{info:3}"
            e_str = "E"
            pbar.set_description(f"{e_str}={e0:.9f}{s2_extra}")

            if save_interval is not None:
                if step % save_interval == 0:
                    fname = prefix + inp.split("/")[-1].replace(".xyz", f"_{basis}_{spin}_ndet{ndet:06}_step{step:06}.h5")
                    save_h5(fname, X._to_nonnormalized_arrays(), mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data=extra_data, postprocess_data=postprocess_data, cart=cart)

        fname = prefix + inp.split("/")[-1].replace(".xyz", f"_{basis}_{spin}_ndet{ndet:06}_step{step:06}.h5")
        save_h5(fname, X._to_nonnormalized_arrays(), mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data=extra_data, postprocess_data=postprocess_data, cart=cart)


if __name__ == "__main__":
    main()
