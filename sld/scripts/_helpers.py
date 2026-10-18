import argparse
import jax
import jax.numpy as jnp
import sys
import time
import numpy as np
from functools import partial
from tqdm.auto import tqdm

from sld.mels import compute_in_blocks
from sld.mels._src.hybrid import ghf as hybrid_ghf
from sld.mels._src.hybrid import uhf as hybrid_uhf
from sld.mels.autobatch import auto_batch_size_from_lowering
from sld.utils.hamiltonian import antisymmetrize_two_body
from sld.utils.hamiltonian.orb_conversion import spin_squared_two_body


def init_seed_and_key(seed):
    if seed is None:
        seed = np.random.randint(2**32)
        if jax.process_count() > 1:
            seed = jax.experimental.multihost_utils.broadcast_one_to_all(seed)
    key = jax.random.PRNGKey(seed)
    return int(seed), key


def checkpoint_filename(prefix, inp, suffix):
    return prefix + inp.split("/")[-1].replace(".z", ".xyz").replace(".xyz", suffix)


def run_training_loop(n_steps, step_fn, describe_fn, save_fn=None, save_interval=None, disable_pbar=False):
    with tqdm(total=n_steps, disable=disable_pbar) as pbar:
        for step in range(1, n_steps + 1):
            step_fn(step)

            # still print output for the first steps, but do not include them in the it/s
            if step < 3:
                pbar.n = step
                pbar.last_print_n = step
                pbar.start_t = pbar.last_print_t = time.time()
                pbar.set_description(describe_fn(step))
                pbar.refresh()
            else:
                pbar.update(1)
                pbar.set_description(describe_fn(step))

            if save_fn is not None and save_interval is not None and step % save_interval == 0:
                save_fn(step)

        if save_fn is not None:
            save_fn(step)


def get_batches(n, batch_size):
    if batch_size is None:
        return [np.arange(n)]
    return [np.arange(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


HYBRID_UHF_BACKENDS = ("hybrid", "hybrid_svd")
BACKEND_HELP = "pairwise computes each determinant pair separately; hybrid caches tensors of the bra across kets (LU with --nonorthogonal, SVD otherwise); hybrid_svd always uses SVD"


def s2_arrays(X, nelec_expected, nao):
    if isinstance(X.nelec, tuple):
        N_u, N_d = nelec_expected
        return {'Enuc': 0.5 * (N_u + N_d) + 0.25 * (N_u - N_d) * (N_u - N_d), 'w': -jnp.einsum("ij,ab->iabj", jnp.eye(nao), jnp.eye(nao))}
    return {'Enuc': 0.75 * X.nelec, 'H2e': antisymmetrize_two_body(jnp.asarray(spin_squared_two_body(nao)))}


def two_electrons(nelec):
    # the orbital variations are linearly dependent, cholesky fails
    return (sum(nelec) if isinstance(nelec, tuple) else nelec) == 2


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _rcond(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def add_hybrid_svd_args(parser):
    parser.add_argument("--hybrid_svd_rcond", type=_rcond, default=1e-10, help="relative singular value cutoff of the hybrid SVD (default: 1e-10)")
    parser.add_argument("--hybrid_svd_max_degree", type=int, choices=range(5), default=4, help="max polynomial degree of the hybrid SVD (default: 4)")


def hybrid_factorization(backend, unstable, max_degree):
    # LU has no polynomial terms
    return ("lu", 0) if unstable and backend == "hybrid" else ("svd", max_degree)


def add_uhf_backend_args(parser, kinds):
    parser.add_argument("--backend", choices=("pairwise",) + HYBRID_UHF_BACKENDS, default=None, help=f"matrix element backend (default: hybrid for UHF, pairwise for GHF); {BACKEND_HELP}")
    for kind in kinds:
        parser.add_argument(f"--hybrid_{kind}_cached_batch_size", type=positive_int, default=None, help=f"number of determinants cached together in the hybrid {kind} (default: automatic)")
        parser.add_argument(f"--hybrid_{kind}_pair_batch_size", type=positive_int, default=None, help=f"number of determinants contracted with each cached batch in the hybrid {kind} (default: the {kind} batch size, else 128)")
    add_hybrid_svd_args(parser)
    parser.add_argument("--independent_spins", action="store_true", help="choose the optimized spin of each determinant independently, instead of a random half up and half down (slower)")


def resolve_uhf_backend(parser, args, kinds, ghf, unstable, nopt):
    nshards = jax.device_count()
    name = args.backend or ("pairwise" if ghf or (nshards != 1 and nshards % 2) else "hybrid")
    hybrid = name in HYBRID_UHF_BACKENDS
    if hybrid and ghf:
        parser.error(f"--backend {name} requires UHF")
    if hybrid and nopt < 2:
        parser.error(f"--backend {name} requires at least 2 optimized determinants")
    if hybrid and nshards != 1 and (nshards % 2 or nopt % 2):
        parser.error(f"--backend {name} requires an even number of devices and of optimized determinants")
    if hybrid and args.independent_spins:
        parser.error("--independent_spins requires --backend pairwise")
    # half of the determinants optimize up, the other half down, if there are at least two per device
    static_spin_blocks = hybrid or (not ghf and not args.independent_spins and nopt % nshards == 0 and nopt // nshards >= 2)
    factorization, max_degree = hybrid_factorization(name, unstable, args.hybrid_svd_max_degree)
    backend = dict(uhf_backend=name, static_spin_blocks=static_spin_blocks, hybrid_factorization=factorization, hybrid_svd_rcond=args.hybrid_svd_rcond, hybrid_svd_max_degree=max_degree)
    for kind in kinds:
        # fall back to --batch_size_grad/--batch_size_hess, then --batch_size
        pair = next(b for b in (getattr(args, f"hybrid_{kind}_pair_batch_size"), getattr(args, f"batch_size_{kind[:4]}", None), args.batch_size, 128) if b is not None)
        if pair <= 0:
            parser.error("batch sizes must be positive")
        backend[f"hybrid_{kind}_cached_batch_size"] = getattr(args, f"hybrid_{kind}_cached_batch_size")
        backend[f"hybrid_{kind}_pair_batch_size"] = pair
    return backend


def check_uhf_backend_hamiltonian(parser, backend, hamiltonian):
    if backend["uhf_backend"] in HYBRID_UHF_BACKENDS and (hamiltonian.get("U", None) is not None or hamiltonian.get("U_onsite", None) is not None):
        parser.error(f"--backend {backend['uhf_backend']} does not support U or U_onsite")


def describe_uhf_backend(backend, kinds, unstable, ghf=False):
    name = backend["uhf_backend"]
    if name not in HYBRID_UHF_BACKENDS:
        print(f"backend: pairwise, {'LU' if unstable else 'SVD'}" + (", half up, half down" if backend["static_spin_blocks"] else ", independent spins" if not ghf else ""))
        return
    for kind in kinds:
        degree = min(backend["hybrid_svd_max_degree"], 3 if kind == "gradient" else 4)
        detail = "LU" if backend["hybrid_factorization"] == "lu" else f"SVD, degree {degree}, rcond {backend['hybrid_svd_rcond']:g}"
        cached = backend[f"hybrid_{kind}_cached_batch_size"]
        print(f"{kind} backend: {name}, {detail}, cached batch {'auto' if cached is None else cached}, pair batch {backend[f'hybrid_{kind}_pair_batch_size']}")


def check_lanczos_args(parser, args):
    if args.lanczos_ncv < 3:
        parser.error("--lanczos_ncv must be at least 3")
    if args.lanczos_tol is not None and args.lanczos_tol <= 0:
        parser.error("--lanczos_tol must be positive")


# optimization and postprocessing options recorded in the h5 file
OPTIMIZE_SETTINGS = ("ghf", "real", "nonorthogonal", "dense_solver", "particle_hole", "orthogao", "sweep", "partialopt", "symm_loss", "start", "project_spin", "project_irrep", "project_lz", "linear_subgroup")
POSTPROCESS_SETTINGS = ("backend", "nonorthogonal", "skipvar", "skipdip", "load_integrals_mo", "project_spin", "projected_energy", "projected_variance", "trdm")


def record_solver_settings(extra_data, args, backend):
    extra_data.update(backend)
    extra_data["lanczos_ncv"] = args.lanczos_ncv
    if args.lanczos_tol is None:
        extra_data.pop("lanczos_tol", None)
    else:
        extra_data["lanczos_tol"] = args.lanczos_tol


def add_backend_args(parser):
    parser.add_argument("--backend", choices=("pairwise",) + HYBRID_UHF_BACKENDS, default="hybrid", help=f"matrix element backend (default: hybrid); {BACKEND_HELP}")
    parser.add_argument("--hybrid_cached_batch_size", type=positive_int, default=1, help="number of determinants whose reduced tensors are cached together (default: 1)")
    parser.add_argument("--hybrid_pair_batch_size", type=positive_int, default=None, help="hybrid pair batch size (default: --batch_size / --batch_size_var, else automatic)")
    add_hybrid_svd_args(parser)
    return parser


def hybrid_kwargs(args, batch_size):
    factorization, max_degree = hybrid_factorization(args.backend, args.nonorthogonal, args.hybrid_svd_max_degree)
    return dict(cached_batch_size=args.hybrid_cached_batch_size, pair_batch_size=args.hybrid_pair_batch_size or batch_size, rcond=args.hybrid_svd_rcond, max_degree=max_degree, factorization=factorization)


def _hybrid_expval(X, Y, arrays, *, squared, max_degree, progress=None, **kwargs):
    module = hybrid_uhf if isinstance(X.nelec, tuple) else hybrid_ghf
    f = module.expval_sq if squared else module.expval
    blocks = f(X, X if Y is None else Y, arrays, progress=progress, max_degree=min(max_degree, 4 if squared else 2), **kwargs)
    return tuple(block[:, None, :, None] for block in blocks)


def tuned_pair_batch(args, lower, n, label, fallback, tree, *static):
    cache = vars(args).setdefault("_tuned_pair_batch", {})
    leaves, treedef = jax.tree.flatten(tree)
    key = (label, n, static, str(treedef), tuple((getattr(x, "shape", ()), str(getattr(x, "dtype", type(x).__name__))) for x in leaves))
    if key not in cache:
        cache[key] = auto_batch_size_from_lowering(lower, n, label=label, fallback=fallback)
    return cache[key]


def evaluate_triangular(evaluate, X, arrays, groups, progress=None):
    ndet = X.ndet
    edges = np.unique(np.linspace(0, ndet, groups + 1).round().astype(int))
    whole = None
    for lo, hi in zip(edges, edges[1:]):
        values = [np.asarray(value).reshape(hi - lo, ndet - lo) for value in evaluate(X[lo:hi], X[lo:], arrays, progress=progress)]
        if whole is None:
            for value in values:
                block = value[:, :hi - lo]
                if np.abs(block - block.conj().T).max() > 1e-8 * max(np.abs(block).max(), 1e-2):
                    raise ValueError("--triangular requires a Hermitian operator, use --no-triangular")
            whole = [np.zeros((ndet, ndet), dtype=value.dtype) for value in values]
        for target, value in zip(whole, values):
            target[lo:hi, lo:] = value
            target[hi:, lo:hi] = value[:, hi - lo:].conj().T
    return tuple(jnp.asarray(target)[:, None, :, None] for target in whole)


def progress_wanted(args):
    setting = getattr(args, "progress", "auto")
    if setting == "off":
        return False
    return setting == "on" or sys.stderr.isatty()


def compute_postprocess_mels(impls, X, arrays, args, *, squared=False, Y=None, label=None):
    batch_size = args.batch_size_var if squared and args.batch_size_var is not None else args.batch_size
    wanted = progress_wanted(args)
    if args.backend == "pairwise":
        return compute_in_blocks(
            impls.expval_sq if squared else impls.expval, X, Y,
            n_results=3 if squared else 2, batch_size=batch_size,
            arrays=arrays, hermitian=Y is None, desc=label if wanted else None,
        )

    kwargs = hybrid_kwargs(args, batch_size)
    if kwargs["pair_batch_size"] is None:
        lower = lambda b: jax.jit(partial(_hybrid_expval, squared=squared, **dict(kwargs, pair_batch_size=b))).lower(X, Y, arrays)
        kwargs["pair_batch_size"] = tuned_pair_batch(
            args, lower, len(X if Y is None else Y), "postprocess H^2" if squared else "postprocess expectation-value",
            1 if squared else 128, (X, Y, arrays), kwargs["cached_batch_size"], kwargs["max_degree"], kwargs["factorization"],
        )
    evaluate = partial(_hybrid_expval, squared=squared, **kwargs)
    with tqdm(total=-(-X.ndet // kwargs["cached_batch_size"]), desc=label or ("H^2" if squared else "mels"), unit="bra", disable=not wanted) as bar:
        progress = (lambda: bar.update(1)) if wanted else None
        if Y is None and args.triangular:
            return evaluate_triangular(evaluate, X, arrays, args.triangular_groups, progress)
        return jax.block_until_ready(evaluate(X, Y, arrays, progress=progress))
