from sld.scripts._env import setup_jax

setup_jax()

import jax
import jax.numpy as jnp

import argparse
from functools import partial

import numpy as np
from jax.sharding import reshard

from sld.optimize import optimize_step_exc
from sld.optimize.exc import compute_s_h_all
from sld.mels import get_impls, static_uhf_spin_pattern
from sld.mels.autobatch import auto_cached_batch_size, resolve_batch_sizes

from sld.slater import GSSD, USSD
from sld.solvers.generalized_eigh import generalized_eigh_lowrank
from sld.mels._src.hybrid import uhf as hybrid_uhf
from sld.utils.io import record_settings, save_h5

from .optimize import add_common_parser_args, setup
from sld.symmetry_projection.kernels import get_mo_trafo
from sld.symmetry_projection.projection_setup import prep_pg_symm
from sld.scripts._helpers import HYBRID_UHF_BACKENDS, init_seed_and_key, checkpoint_filename, run_training_loop, add_uhf_backend_args, resolve_uhf_backend, check_uhf_backend_hamiltonian, describe_uhf_backend, check_lanczos_args, record_solver_settings, two_electrons, OPTIMIZE_SETTINGS

HYBRID_KINDS = ("gradient", "hessian")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inp", type=str, help="input .xyz file")
    parser.add_argument("basis", type=str, help="basis")
    parser.add_argument("n_steps", type=int, help="number of iterations")
    parser.add_argument("nstates", type=int, help="number of states")
    parser.add_argument("nper", type=int, help="number of determinants per state")
    add_common_parser_args(parser)
    parser.add_argument("--spin", type=int, default=0, help="spin (pyscf). This is 2*M = 2*<Sz>. Defaults to 0.")
    parser.add_argument("--ghf", action="store_true", help="use generalized hartree-fock determinants; if not set the code uses UHF")
    parser.add_argument("--particle_hole", action="store_true", help="apply a particle hole transformation to the hamiltonian")
    parser.add_argument("--orthogao", action="store_true", help="diagonalize the overlap matrix to get orthogonal orbitals, and do NOT run a HF calculation or use MO")
    parser.add_argument("--cart", action="store_true", help="use cart orbitals instead of sph")
    parser.add_argument("--sweep", action="store_true", help="optimize the states in sweeps; if not set we select uniformly at every step")
    parser.add_argument("--s2_loss", type=float, required=False, help="add a S^2 penalty term to the Hamiltonian, scaled by the value of this argument")
    parser.add_argument("--symm_loss", type=str, required=False, help="add weighted point-group operators to the Hamiltonian, given as \"w1 op1 w2 op2 ...\" (the available operator names are printed when the option is used)")
    parser.add_argument("--batch_size_expval", type=int, default=None, help="batch size for the expectation values (default: --batch_size, else automatic)")
    parser.add_argument("--batch_size_grad", type=int, default=None, help="batch size for the gradient (default: --batch_size, else automatic)")
    parser.add_argument("--batch_size_hess", type=int, default=None, help="batch size for the Hessian (default: --batch_size, else automatic)")
    add_uhf_backend_args(parser, HYBRID_KINDS)
    parser.add_argument("--uniform_spin_count", action="store_true", help="random number of spin up updates per step; requires --backend pairwise")

    args = parser.parse_args()

    inp = args.inp
    basis = args.basis
    n_steps = args.n_steps
    nstates = args.nstates
    nper = args.nper
    seed = args.seed
    start = args.start
    prefix = args.prefix
    cplx = not args.real
    batch_size = args.batch_size
    batch_size_expval = args.batch_size_expval
    batch_size_grad = args.batch_size_grad
    batch_size_hess = args.batch_size_hess
    lanczos_ncv = args.lanczos_ncv
    lanczos_tol = args.lanczos_tol
    unstable = args.nonorthogonal
    save_interval = args.save_interval
    spin = args.spin  # 2 * Sz
    ghf = args.ghf
    particle_hole = args.particle_hole
    orthogao = args.orthogao
    cart = args.cart
    sweep = args.sweep
    nfreeze = args.nfreeze

    s2_loss = args.s2_loss

    if cplx:
        dtype = complex
    else:
        dtype = float

    if nper % jax.device_count() != 0:
        raise ValueError("number of determinants per state needs to be a multiple of the number of jax devices")
    if args.dense_solver:
        raise NotImplementedError("--dense_solver")
    check_lanczos_args(parser, args)
    if args.uniform_spin_count:
        if ghf or args.backend != "pairwise":
            parser.error("--uniform_spin_count requires UHF and --backend pairwise")
        args.independent_spins = True
    backend = resolve_uhf_backend(parser, args, HYBRID_KINDS, ghf, unstable, nper)
    hybrid = backend["uhf_backend"] in HYBRID_UHF_BACKENDS
    if hybrid and nstates < 2:
        parser.error("hybrid UHF backends require at least two states")

    hamiltonian, X_start, mo_coeff, extra_data, name, xyzstr, no, nelec = setup(inp, basis, spin, cart, start, s2_loss, unstable, ghf, particle_hole, orthogao, nfreeze, charge=args.charge)
    if two_electrons(nelec):
        raise NotImplementedError("two electrons need the dense solver of sld-optimize")

    if args.symm_loss is not None:
        symops, *_ = prep_pg_symm(xyzstr, basis, cart=cart)
        mo_trafos = {}
        for k, ao_trafo in symops.items():
            ao_coeff = np.eye(mo_coeff.shape[0])
            mo_trafos[k] = get_mo_trafo(ao_trafo, mo_coeff, ao_coeff)
        print('available symm operators:', list(mo_trafos.keys()))

        sl = args.symm_loss.split(' ')
        U = 0
        for w, k in zip(sl[::2], sl[1::2]):
            w = float(w)
            print('adding', w, k)
            U = U + w * mo_trafos[k]
        if not np.allclose(U, 0):
            hamiltonian['U'] = U, U

    extra_data["nper"] = int(nper)
    extra_data["nstates"] = int(nstates)
    check_uhf_backend_hamiltonian(parser, backend, hamiltonian)
    extra_data["uniform_spin_count"] = args.uniform_spin_count
    impls = get_impls(ghf, unstable)
    describe_uhf_backend(backend, HYBRID_KINDS, unstable, ghf)

    seed, key = init_seed_and_key(seed)
    extra_data["seed"] = seed

    key, k0 = jax.random.split(key)
    ndet = nper * nstates

    if isinstance(nelec, tuple):
        X = USSD.random(k0, ndet, nelec, no, dtype)
    else:
        X = GSSD.random(k0, ndet, nelec, no, dtype)

    if X_start is not None:
        nstates_start, nper_start = X_start.coeffs.shape
        if (nstates_start > nstates) or (nper_start > nper):
            raise ValueError('start file has more states or determinants than this simulation', (nstates, nper), (nstates_start, nper_start))
        # unflatten
        X = jax.tree.map(lambda x: x.reshape(nstates, nper, *x.shape[1:]), X)
        X = X.replace(coeffs=X.coeffs.at[:nstates_start, nper_start:].set(0))
        # copy starting dets into X
        X = jax.tree.map(lambda x, y: x.at[:nstates_start, :nper_start].set(y), X, X_start)
        # flatten
        X = jax.tree.map(partial(jax.lax.collapse, start_dimension=0, stop_dimension=2), X)

    X_opt_probe = X[:nper].choose_random_orbitals_to_optimize(jax.random.fold_in(k0, 1), optimize_first_only=False)
    X_fix_probe = X[nper:]
    if hybrid:
        X_opt_probe = X_opt_probe.replace(s=static_uhf_spin_pattern(X_opt_probe.ndet, nshards=1))
        hess_kwargs = dict(rcond=backend["hybrid_svd_rcond"], factorization=backend["hybrid_factorization"], max_degree=min(backend["hybrid_svd_max_degree"], 4), pair_batch_size=backend["hybrid_hessian_pair_batch_size"])
        grad_kwargs = dict(hess_kwargs, max_degree=min(backend["hybrid_svd_max_degree"], 3), pair_batch_size=backend["hybrid_gradient_pair_batch_size"])
        if jax.device_count() == 1:
            lower_hessian = lambda candidate: hybrid_uhf.hessian.lower(X_opt_probe, arrays=hamiltonian, cached_batch_size=candidate, **hess_kwargs)
            lower_gradient = lambda candidate: hybrid_uhf.gradient.lower(X_opt_probe, X_fix_probe, arrays=hamiltonian, cached_batch_size=candidate, **grad_kwargs)
        else:
            X_opt_bra = reshard(X_opt_probe, jax.P("i"))
            lower_hessian = lambda candidate: hybrid_uhf.hessian_sharded.lower(X_opt_bra, arrays=hamiltonian, Y=X_opt_probe, cached_batch_size=candidate, **hess_kwargs)
            lower_gradient = lambda candidate: hybrid_uhf.gradient_sharded.lower(X_opt_bra, X_fix_probe, arrays=hamiltonian, cached_batch_size=candidate, **grad_kwargs)
        if backend["hybrid_hessian_cached_batch_size"] is None:
            backend["hybrid_hessian_cached_batch_size"] = auto_cached_batch_size(lower_hessian, len(X_opt_probe) // jax.device_count(), label="Hessian")
        if backend["hybrid_gradient_cached_batch_size"] is None:
            backend["hybrid_gradient_cached_batch_size"] = auto_cached_batch_size(lower_gradient, len(X_opt_probe) // jax.device_count(), label="gradient")
    if backend["hybrid_hessian_cached_batch_size"] is None:
        backend["hybrid_hessian_cached_batch_size"] = 16
    if backend["hybrid_gradient_cached_batch_size"] is None:
        backend["hybrid_gradient_cached_batch_size"] = 1
    record_solver_settings(extra_data, args, backend)
    record_settings(extra_data, args, OPTIMIZE_SETTINGS)

    batch_size_expval, batch_size_grad, batch_size_hess = resolve_batch_sizes(
        batch_size,
        backend["hybrid_hessian_pair_batch_size"] if hybrid else batch_size_expval,
        backend["hybrid_gradient_pair_batch_size"] if hybrid else batch_size_grad,
        backend["hybrid_hessian_pair_batch_size"] if hybrid else batch_size_hess,
        impls=impls,
        X_expval=X,
        Y_expval=X,
        X_grad=X_opt_probe,
        Y_grad=X_fix_probe,
        X_hess=X_opt_probe,
        Y_hess=X_opt_probe,
        arrays=hamiltonian,
    )
    print(f"pair batches: expval={batch_size_expval}, grad={batch_size_grad}, hess={batch_size_hess}")

    # cache for the overlap and matrix elements between the different states
    X, S_all, H_all = compute_s_h_all(impls, X, hamiltonian, nstates, batch_size if batch_size_expval is None else batch_size_expval)

    eigvals = []
    energy_vals = []
    postprocess_data = {}
    last = {}

    def step_fn(step):
        nonlocal key, X, S_all, H_all
        key, k = jax.random.split(key)
        X, e0, info, S_all, H_all = optimize_step_exc(
            k,
            X,
            hamiltonian,
            impls,
            nper=nper,
            batch_size=batch_size,
            S_all=S_all,
            H_all=H_all,
            j_opt=step % nstates if sweep else None,
            keep_order=sweep,
            batch_size_expval=batch_size_expval,
            batch_size_grad=batch_size_grad,
            batch_size_hess=batch_size_hess,
            lanczos_ncv=lanczos_ncv,
            lanczos_tol=lanczos_tol,
            uniform_spin_count=args.uniform_spin_count,
            **backend,
        )

        # this is just a small nstates x nstates problem, so we solve it serially using dense eigendecomp
        # consider skipping if we use a very large number of states
        e_vals, _ = generalized_eigh_lowrank(H_all, S_all)
        if jnp.isnan(e0).any():
            raise ValueError('observed nan in eigvals')
        eigvals.append(np.array(e0).reshape(()).tolist())
        energy_vals.append(np.array(e_vals).tolist())
        extra_data["energy_vals"] = energy_vals
        last["e0"], last["e_vals"] = e0, e_vals

    def describe_fn(step):
        e0, e_vals = last["e0"], last["e_vals"]
        s2_extra = ", e: [" + "".join([f"{e:.5f}, " for e in e_vals[: min(nstates, 8)]])
        if nstates > 8:
            s2_extra += "..."
        else:
            s2_extra = s2_extra[:-2]
        return f"L={e0:.9f}{s2_extra}"

    def save_fn(step):
        fname = checkpoint_filename(prefix, inp, f"_{basis}_{spin}_nstates{nstates:04}_nper{nper:04}_step{step:06}.h5")
        save_h5(
            fname,
            X._to_nonnormalized_arrays(),
            mo_coeff,
            eigvals,
            xyzstr,
            basis,
            spin,
            name,
            extra_data=extra_data,
            postprocess_data=postprocess_data,
            cart=cart,
            nfreeze=nfreeze,
        )

    run_training_loop(
        n_steps, step_fn, describe_fn, save_fn=save_fn, save_interval=save_interval, disable_pbar=jax.process_index() > 0
    )


if __name__ == "__main__":
    main()
