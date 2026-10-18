from sld.scripts._env import setup_jax

setup_jax()

import jax
import jax.numpy as jnp
import argparse
import numpy as np
from jax.sharding import reshard

from sld.mels._src.hybrid import uhf as hybrid_uhf
from sld.utils.io import record_settings, save_h5

from sld.mels import get_impls, static_uhf_spin_pattern
from sld.mels.autobatch import auto_cached_batch_size, resolve_batch_sizes
from .optimize import add_common_parser_args, setup

from sld.slater import GSSD, USSD


from sld.optimize import optimize_step_partial
from sld.scripts._helpers import HYBRID_UHF_BACKENDS, init_seed_and_key, checkpoint_filename, run_training_loop, add_uhf_backend_args, resolve_uhf_backend, check_uhf_backend_hamiltonian, describe_uhf_backend, check_lanczos_args, record_solver_settings, two_electrons, OPTIMIZE_SETTINGS

HYBRID_KINDS = ("gradient", "hessian")


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
    parser.add_argument("--orthogao", action="store_true", help="diagonalize the overlap matrix to get orthogonal orbitals, and do NOT run a HF calculation or use MO")
    parser.add_argument("--cart", action="store_true", help="use cart orbitals instead of sph")
    parser.add_argument("--partialopt", type=int, required=True, help="optimize only a subset of determinants, specify how many")
    parser.add_argument("--batch_size_expval", type=int, default=None, help="batch size for the expectation values (default: --batch_size, else automatic)")
    parser.add_argument("--batch_size_grad", type=int, default=None, help="batch size for the gradient (default: --batch_size, else automatic)")
    parser.add_argument("--batch_size_hess", type=int, default=None, help="batch size for the Hessian (default: --batch_size, else automatic)")
    add_uhf_backend_args(parser, HYBRID_KINDS)

    args = parser.parse_args()

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
    batch_size_expval = args.batch_size_expval
    batch_size_grad = args.batch_size_grad
    batch_size_hess = args.batch_size_hess
    lanczos_ncv = args.lanczos_ncv
    lanczos_tol = args.lanczos_tol
    save_interval = args.save_interval
    partialopt = args.partialopt
    particle_hole = args.particle_hole
    s2_loss = args.s2_loss
    orthogao = args.orthogao
    cart = args.cart
    cholesky = not args.dense_solver
    nfreeze = args.nfreeze

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
    check_lanczos_args(parser, args)
    if spin != 0 and not ghf:
        raise NotImplementedError("partial optimization with nelec[0] != nelec[1]")
    backend = resolve_uhf_backend(parser, args, HYBRID_KINDS, ghf, unstable, partialopt)
    hybrid = backend["uhf_backend"] in HYBRID_UHF_BACKENDS

    hamiltonian, X_start, mo_coeff, extra_data, name, xyzstr, no, nelec = setup(inp, basis, spin, cart, start, s2_loss, unstable, ghf, particle_hole, orthogao, nfreeze, charge=args.charge)
    if two_electrons(nelec):
        raise NotImplementedError("two electrons need the dense solver of sld-optimize")

    check_uhf_backend_hamiltonian(parser, backend, hamiltonian)
    impls = get_impls(ghf, unstable)
    describe_uhf_backend(backend, HYBRID_KINDS, unstable, ghf)

    seed, key = init_seed_and_key(seed)
    extra_data["seed"] = seed

    key, k0 = jax.random.split(key)
    print("initializing determinants")
    if X_start is not None:
        # pad with random determinants
        X = X_start.pad(k0, ndet - len(X_start))
    elif isinstance(nelec, tuple):
        X = USSD.random(k0, ndet, nelec, no, dtype)
    else:
        X = GSSD.random(k0, ndet, nelec, no, dtype)

    Xf_probe = X[:partialopt].choose_random_orbitals_to_optimize(jax.random.fold_in(k0, 1), optimize_first_only=True)
    Xl_probe = X[partialopt:]
    if hybrid:
        Xf_probe = Xf_probe.replace(s=static_uhf_spin_pattern(Xf_probe.ndet, nshards=1))
        hess_kwargs = dict(rcond=backend["hybrid_svd_rcond"], factorization=backend["hybrid_factorization"], max_degree=min(backend["hybrid_svd_max_degree"], 4), pair_batch_size=backend["hybrid_hessian_pair_batch_size"])
        grad_kwargs = dict(hess_kwargs, max_degree=min(backend["hybrid_svd_max_degree"], 3), pair_batch_size=backend["hybrid_gradient_pair_batch_size"])
        if jax.device_count() == 1:
            lower_hessian = lambda candidate: hybrid_uhf.hessian.lower(Xf_probe, arrays=hamiltonian, cached_batch_size=candidate, **hess_kwargs)
            lower_gradient = lambda candidate: hybrid_uhf.gradient.lower(Xf_probe, Xl_probe, arrays=hamiltonian, cached_batch_size=candidate, **grad_kwargs)
        else:
            Xf_bra = reshard(Xf_probe, jax.P("i"))
            lower_hessian = lambda candidate: hybrid_uhf.hessian_sharded.lower(Xf_bra, arrays=hamiltonian, Y=Xf_probe, cached_batch_size=candidate, **hess_kwargs)
            lower_gradient = lambda candidate: hybrid_uhf.gradient_sharded.lower(Xf_bra, Xl_probe, arrays=hamiltonian, cached_batch_size=candidate, **grad_kwargs)
        if backend["hybrid_hessian_cached_batch_size"] is None:
            backend["hybrid_hessian_cached_batch_size"] = auto_cached_batch_size(lower_hessian, len(Xf_probe) // jax.device_count(), label="Hessian")
        if backend["hybrid_gradient_cached_batch_size"] is None:
            backend["hybrid_gradient_cached_batch_size"] = auto_cached_batch_size(lower_gradient, len(Xf_probe) // jax.device_count(), label="gradient")
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
        X_expval=Xl_probe,
        Y_expval=Xl_probe,
        X_grad=Xf_probe,
        Y_grad=Xl_probe,
        X_hess=Xf_probe,
        Y_hess=Xf_probe,
        arrays=hamiltonian,
    )
    print(f"pair batches: expval={batch_size_expval}, grad={batch_size_grad}, hess={batch_size_hess}")

    S_all = H_all = None
    eigvals = []
    postprocess_data = {}
    last = {}

    def step_fn(step):
        nonlocal key, X, S_all, H_all
        key, k = jax.random.split(key)
        if S_all is not None:
            c = jnp.abs(X.coeffs * S_all.diagonal())
            # if some dets have almost zero weight we choose the ones with small weight more often for the orbital optimization
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
            ndet_partial=partialopt,
            batch_size=batch_size,
            S_all=S_all,
            H_all=H_all,
            p_prob=p_prob,
            cholesky=cholesky,
            batch_size_expval=batch_size_expval,
            batch_size_grad=batch_size_grad,
            batch_size_hess=batch_size_hess,
            lanczos_ncv=lanczos_ncv,
            lanczos_tol=lanczos_tol,
            **backend,
        )
        eigvals.append(np.array(e0).reshape(()).tolist())
        last["e0"], last["info"] = e0, info

    def describe_fn(step):
        e0, info = last["e0"], last["info"]
        cmin = jnp.abs(X.coeffs * S_all.diagonal()).min().tolist()  # minimum norm of any det of the normalized state
        return f"E={e0:.9f} cmin:{cmin:0.1e} iter:{info:3}"

    def save_fn(step):
        fname = checkpoint_filename(prefix, inp, f"_{basis}_{spin}_ndet{ndet:06}_step{step:06}.h5")
        save_h5(fname, X._to_nonnormalized_arrays(), mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data=extra_data, postprocess_data=postprocess_data, cart=cart, nfreeze=nfreeze)

    run_training_loop(n_steps, step_fn, describe_fn, save_fn=save_fn, save_interval=save_interval)


if __name__ == "__main__":
    main()
