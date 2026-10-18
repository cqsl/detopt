from sld.scripts._env import setup_jax

setup_jax()

import jax
import jax.numpy as jnp
import argparse
import numpy as np
from jax.sharding import reshard

from sld.slater import GSSD, USSD
from sld.optimize.full import optimize_step
from sld.utils.hamiltonian import orbitals_pyscf, prepare_hamiltonian
from sld.utils.hamiltonian import ao2mo, freeze_core
from sld.mels import compute_mels_static_uhf_spin_blocks, get_impls, static_uhf_spin_pattern
from sld.mels.autobatch import auto_batch_size, auto_batch_size_from_lowering, auto_cached_batch_size, check_full_hessian_output_memory
from sld.mels._src.hybrid import uhf as hybrid_uhf
from sld.utils.io import load_h5, record_settings, save_h5
from sld.scripts._helpers import HYBRID_UHF_BACKENDS, init_seed_and_key, checkpoint_filename, run_training_loop, add_uhf_backend_args, resolve_uhf_backend, check_uhf_backend_hamiltonian, describe_uhf_backend, check_lanczos_args, record_solver_settings, two_electrons, OPTIMIZE_SETTINGS


HYBRID_KINDS = ("hessian",)


def build_hamiltonian(xyzstr, basis, spin, cart, s2_loss, unstable, ghf, particle_hole, orthogao, nfreeze, mo_coeff=None, symmetry=True, charge=0):

    if not unstable and not ghf:
        print("WARNING: using the slow SVD code; consider using --nonorthogonal if you think it is safe to assume that the determinants will be not orthogonal")

    nao, nelec, mo_coeff, enuc, h1e_ao, h2e_ao, *_ = orbitals_pyscf(xyzstr, basis, spin, mo_coeff=mo_coeff, run_hf=not orthogao, cart=cart, symmetry=symmetry, charge=charge)
    print(f"nelec: {nelec}; nao: {nao}")

    if particle_hole:
        nelec = nao - nelec[0], nao - nelec[1]

    if ghf:
        nelec = sum(nelec)
        no = 2 * nao
    else:
        no = nao
    h1e_mo = ao2mo(h1e_ao, mo_coeff)
    h2e_mo = ao2mo(h2e_ao, mo_coeff)
    del h1e_ao
    del h2e_ao

    if nfreeze is not None:
        if particle_hole:
            raise NotImplementedError("--nfreeze with --particle_hole")
        print("frozen core calculation")
        enuc, h1e_mo, h2e_mo = freeze_core(enuc, h1e_mo, h2e_mo, nfreeze=nfreeze)
        if ghf:
            nelec = nelec - nfreeze
        else:
            nelec = nelec[0] - nfreeze // 2, nelec[1] - nfreeze // 2
        no = h1e_mo.shape[0]
    print(f"correlating {nelec} electrons in {no} orbitals")

    print("preparing hamiltonian arrays")
    hamiltonian = prepare_hamiltonian(enuc, h1e_mo, h2e_mo, ghf, particle_hole=particle_hole)
    del h1e_mo
    del h2e_mo

    if s2_loss is not None:
        assert not ghf
        # we would not need the constant term just for the loss,
        # but this way we can subtract <S^2> later to get the energy
        N_d = nelec[0]
        N_u = nelec[1]
        hamiltonian['Enuc'] = hamiltonian['Enuc'] + s2_loss * (0.5 * (N_u + N_d) + 0.25 * (N_u - N_d) * (N_u - N_d))
        I = jnp.eye(no)
        hamiltonian['w'] = hamiltonian['w'] - s2_loss * jnp.einsum("jk,il->ijkl", I, I)

    return hamiltonian, mo_coeff, no, nelec


def setup(inp, basis, spin, cart, start, s2_loss, unstable, ghf, particle_hole, orthogao, nfreeze, symmetry=True, charge=0):
    with open(inp, "r") as f:
        xyzstr = f.read()

    name = inp.split("/")[-1]
    extra_data = {}
    if s2_loss is not None:
        extra_data["s2_loss"] = float(s2_loss)

    if start is not None:
        Xs_start, mo_coeff, _eigvals, _xyzstr, _basis, _spin, _name, extra_data, _, cart, _nfreeze = load_h5(start)
        if not (_basis == basis and _spin == spin and _name == name and _nfreeze == nfreeze and int(extra_data.get("charge", 0)) == charge):
            raise NotImplementedError("--start with a different basis, spin, charge, input file or --nfreeze")
    else:
        Xs_start = None
        mo_coeff = None

    hamiltonian, mo_coeff, no, nelec = build_hamiltonian(
        xyzstr, basis, spin, cart, s2_loss, unstable,
        ghf, particle_hole, orthogao, nfreeze, mo_coeff=mo_coeff,
        symmetry=symmetry, charge=charge,
    )
    # only stored for ions
    extra_data.pop("charge", None)
    if charge:
        extra_data["charge"] = int(charge)

    if Xs_start is not None:
        if isinstance(Xs_start, tuple):
            coeffs_start = jnp.ones(Xs_start[0].shape[0], dtype=Xs_start[0].dtype)
            X_start = USSD(*Xs_start, coeffs_start).normalize_orbs()
        else:
            coeffs_start = jnp.ones(Xs_start.shape[0], dtype=Xs_start.dtype)
            X_start = GSSD(Xs_start, coeffs_start).normalize_orbs()
    else:
        X_start = None
    if (nper := extra_data.get('nper', None)) is not None:
        nper = int(nper)
        nstates = X_start.ndet//nper
        # excited-state checkpoint: keep the (nstates, nper) layout, which only sld-optimize-exc reads
        X_start = jax.tree.map(lambda x: x.reshape(nstates, nper, *x.shape[1:]), X_start)

    return hamiltonian, X_start, mo_coeff, extra_data, name, xyzstr, no, nelec


def add_common_parser_args(parser):
    parser.add_argument("--seed", type=int, required=False, help="random seed (an integer)")
    parser.add_argument("--start", type=str, required=False, help="determinant initialization (h5 file from a previous run), needs to contain <= ndet determinants")
    parser.add_argument("--prefix", type=str, default="./", help="prefix to append to the output filename (path and/or a name)")
    parser.add_argument("--real", action="store_true", help="use real parameters; if not set defaults to complex parameters")
    parser.add_argument("--batch_size", type=int, required=False, help="batch size for the matrix elements (default: automatic)")
    parser.add_argument("--nonorthogonal", action="store_true", help="assume determinants are non-orthogonal; uses faster code path")
    parser.add_argument("--save_interval", type=int, required=False, help="every how many steps to save the state (default: only at the end)")
    parser.add_argument("--dense_solver", action="store_true", help="use dense eigh to solve the eigenvalue problem instead of Cholesky+Lanczos")
    parser.add_argument("--nfreeze", type=int, required=False, help="frozencore: how many electrons to freeze in total (half up half down)")
    parser.add_argument("--charge", type=int, default=0, help="total charge of the molecule (default: 0)")
    parser.add_argument("--lanczos_ncv", type=int, default=255, help="number of Lanczos Krylov vectors per restart (default: 255; only used by the Cholesky+Lanczos solver)")
    parser.add_argument("--lanczos_tol", type=float, default=None, help="Lanczos residual tolerance (default: machine precision; only used by the Cholesky+Lanczos solver)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inp", type=str, help="input .xyz file")
    parser.add_argument("basis", type=str, help="basis")
    parser.add_argument("n_steps", type=int, help="number of iterations")
    parser.add_argument("ndet", type=int, help="number of determinants")
    add_common_parser_args(parser)
    parser.add_argument("--spin", type=int, default=0, help="spin (pyscf). This is 2*M = 2*<Sz>. Defaults to 0.")
    parser.add_argument("--ghf", action="store_true", help="use generalized hartree-fock determinants; if not set the code uses UHF")
    add_uhf_backend_args(parser, HYBRID_KINDS)
    parser.add_argument("--particle_hole", action="store_true", help="apply a particle hole transformation to the hamiltonian")
    parser.add_argument("--s2_loss", type=float, required=False, help="add a S^2 penalty term to the Hamiltonian, scaled by the value of this argument")
    parser.add_argument("--orthogao", action="store_true", help="diagonalize the overlap matrix to get orthogonal orbitals, and do NOT run a HF calculation or use MO")
    parser.add_argument("--cart", action="store_true", help="use cart orbitals instead of sph")

    args = parser.parse_args()

    n_steps = args.n_steps
    ndet = args.ndet
    seed = args.seed
    basis = args.basis
    inp = args.inp
    prefix = args.prefix
    spin = args.spin  # 2 * Sz
    ghf = args.ghf
    lanczos_ncv = args.lanczos_ncv
    lanczos_tol = args.lanczos_tol
    start = args.start
    unstable = args.nonorthogonal
    cplx = not args.real
    batch_size = args.batch_size
    save_interval = args.save_interval
    particle_hole = args.particle_hole
    s2_loss = args.s2_loss
    orthogao = args.orthogao
    cart = args.cart
    cholesky = not args.dense_solver
    nfreeze = args.nfreeze

    backend = resolve_uhf_backend(parser, args, HYBRID_KINDS, ghf, unstable, ndet)
    check_lanczos_args(parser, args)
    if cplx:
        dtype = complex
    else:
        dtype = float

    if ndet % jax.device_count() != 0:
        raise ValueError("number of determinants needs to be a multiple of the number of jax devices")

    hamiltonian, X_start, mo_coeff, extra_data, name, xyzstr, no, nelec = setup(inp, basis, spin, cart, start, s2_loss, unstable, ghf, particle_hole, orthogao, nfreeze, charge=args.charge)
    check_uhf_backend_hamiltonian(parser, backend, hamiltonian)
    if cholesky and two_electrons(nelec):
        print("two electrons, using the dense solver")
        cholesky = False
        args.dense_solver = True
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

    try:
        check_full_hessian_output_memory(X.ndet, X.n_orbitals, X.dtype.itemsize, jax.device_count())
    except MemoryError as error:
        parser.exit(1, f"error: {error}\n")

    if backend["hybrid_hessian_cached_batch_size"] is None:
        if backend["uhf_backend"] in HYBRID_UHF_BACKENDS:
            X_cached_probe = X.choose_random_orbitals_to_optimize(jax.random.fold_in(k0, 2), optimize_first_only=True).replace(s=static_uhf_spin_pattern(X.ndet, nshards=1))
            kwargs = dict(rcond=backend["hybrid_svd_rcond"], max_degree=backend["hybrid_svd_max_degree"], factorization=backend["hybrid_factorization"], pair_batch_size=backend["hybrid_hessian_pair_batch_size"])
            if jax.device_count() == 1:
                lower = lambda candidate: hybrid_uhf.hessian.lower(X_cached_probe, hamiltonian, cached_batch_size=candidate, **kwargs)
            else:
                X_cached_bra = reshard(X_cached_probe, jax.P("i"))
                lower = lambda candidate: hybrid_uhf.hessian_sharded.lower(X_cached_bra, arrays=hamiltonian, Y=X_cached_probe, cached_batch_size=candidate, **kwargs)
            backend["hybrid_hessian_cached_batch_size"] = auto_cached_batch_size(lower, X.ndet // jax.device_count(), label="Hessian")
        else:
            backend["hybrid_hessian_cached_batch_size"] = 16

    hybrid = backend["uhf_backend"] in HYBRID_UHF_BACKENDS
    if batch_size is None and not hybrid:
        X_hess_probe = X.choose_random_orbitals_to_optimize(jax.random.fold_in(k0, 1), optimize_first_only=True)
        if backend["static_spin_blocks"]:
            X_hess_probe = X_hess_probe.replace(s=static_uhf_spin_pattern(X.ndet))
            hessian_driver = compute_mels_static_uhf_spin_blocks
            if jax.device_count() == 1:
                nup = (X.ndet + 1) // 2
                ndown = X.ndet - nup
                hessian_npairs = max(nup * nup, nup * ndown, ndown * ndown)
            else:
                hessian_npairs = X.ndet * X.ndet
            batch_size = auto_batch_size_from_lowering(lambda candidate: hessian_driver.lower(impls.hess_spin_blocks, X_hess_probe, arrays=hamiltonian, batch_size=candidate, nshards=jax.device_count()), hessian_npairs, label="Hessian", min_batch=max(1, (X.ndet + 1) // 2))
        else:
            batch_size = auto_batch_size(impls.hess, X_hess_probe, X_hess_probe, hamiltonian, label="Hessian")
    batch_size_hess = backend["hybrid_hessian_pair_batch_size"] if hybrid else batch_size
    print(f"pair batch: hess={batch_size_hess}")
    extra_data["batch_size_hess"] = batch_size_hess
    record_solver_settings(extra_data, args, backend)
    record_settings(extra_data, args, OPTIMIZE_SETTINGS)

    eigvals = []
    postprocess_data = {}
    last = {}

    def step_fn(step):
        nonlocal key, X
        key, k = jax.random.split(key)
        X, e0, info = optimize_step(
            k,
            X,
            hamiltonian,
            impls,
            batch_size=batch_size,
            cholesky=cholesky,
            lanczos_ncv=lanczos_ncv,
            lanczos_tol=lanczos_tol,
            **backend,
        )
        eigvals.append(np.array(e0).reshape(()).tolist())
        last["e0"], last["info"] = e0, info

    def describe_fn(step):
        e0, info = last["e0"], last["info"]
        s2_extra = f" cmin:{jnp.abs(X.coeffs).min().tolist():0.1e}"
        if info is not None:
            s2_extra += f" iter:{info:3}"
        return f"E={e0:.9f}{s2_extra}"

    def save_fn(step):
        fname = checkpoint_filename(prefix, inp, f"_{basis}_{spin}_ndet{ndet:06}_step{step:06}.h5")
        save_h5(fname, X._to_nonnormalized_arrays(), mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data=extra_data, postprocess_data=postprocess_data, cart=cart, nfreeze=nfreeze)

    run_training_loop(n_steps, step_fn, describe_fn, save_fn=save_fn, save_interval=save_interval)


if __name__ == "__main__":
    main()
