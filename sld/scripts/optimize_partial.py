import jax

jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, set_mesh, AxisType

mesh = Mesh(jax.devices(), "i", AxisType.Explicit)
set_mesh(mesh)


import jax.numpy as jnp
import argparse
import numpy as np
from tqdm.auto import tqdm

from sld.utils.io import save_h5

from sld.mels import get_impls
from .optimize import add_common_parser_args, setup

from sld.slater import GSSD, USSD


from sld.optimize import optimize_step_partial


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
    parser.add_argument("--partialopt", type=int, required=True, help="optimize only a subset of determinants, specify how many")

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
    partialopt = args.partialopt
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
    if partialopt is not None and partialopt % jax.device_count() != 0:
        raise ValueError("number of determinants optimized per step needs to be a multiple of the number of jax devices")
    if partialopt >= ndet or ndet % partialopt != 0:
        raise ValueError("number of determinants optimized needs to be a divisor of ndet; use sld-optimize for full optimization")

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

    S_all = H_all = None
    eigvals = []
    postprocess_data = {}

    with tqdm(total=n_steps) as pbar:
        for step in range(1, n_steps + 1):
            key, k = jax.random.split(key)
            if S_all is not None:
                c = jnp.abs(X.coeffs * S_all.diagonal())
                if c.min() < 1e-6:  # the whole state is normalized, so here we don't bother putting the max
                    p_prob = -jnp.log(c)
                    p_prob = p_prob / p_prob.sum()
                else:
                    p_prob = None
            else:
                p_prob = None

            X, e0, info, S_all, H_all = optimize_step_partial(
                k,
                X,
                hamiltonian,
                impls,
                mo_coeff=mo_coeff_later,
                ndet_partial=partialopt,
                batch_size=batch_size,
                S_all=S_all,
                H_all=H_all,
                p_prob=p_prob,
                cholesky=cholesky,
            )

            eigvals.append(np.array(e0).reshape(()).tolist())
            pbar.update(1)

            s2_extra = ""
            cmin = jnp.abs(X.coeffs * S_all.diagonal()).min().tolist()  # minimum norm of any det of the normalized state
            s2_extra = s2_extra + f" cmin:{cmin:0.1e}"
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
