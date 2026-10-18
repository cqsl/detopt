from sld.scripts._env import setup_jax

setup_jax()

from jax.tree_util import Partial
from tqdm.auto import tqdm
from sld.scripts._helpers import positive_int, hybrid_kwargs, progress_wanted, tuned_pair_batch, compute_in_blocks, get_batches

from sld.symmetry_projection import get_projected_expval_impl
from sld.mels._src.hybrid import mels_proj as hybrid_projection
from sld.symmetry_projection.kernels import lz_rot_trafos

from sld.symmetry_projection.projection_setup import prep_symm_trafos


_bar = [None]


def _tick():
    _bar[0].update(1)


def compute_projected_overlap(X, args, weights, trafos, *, ghf_operator=False, arrays=None, squared=False, progress=None):
    ghf = not isinstance(X.nelec, tuple)
    if args.backend == "pairwise":
        if squared:
            raise NotImplementedError("the projected <H^2> has no pairwise implementation; use --backend hybrid")
        expval = get_projected_expval_impl(
            ghf, args.nonorthogonal, ghf_operator=ghf_operator,
        ).expval
    else:
        if ghf:
            expval = hybrid_projection.expval_sq_proj_ghf if squared else hybrid_projection.expval_proj_ghf
        elif ghf_operator:
            expval = hybrid_projection.expval_sq_proj_uhf_via_ghf if squared else hybrid_projection.expval_proj_uhf_via_ghf
        else:
            expval = hybrid_projection.expval_sq_proj_uhf if squared else hybrid_projection.expval_proj_uhf
        kwargs = hybrid_kwargs(args, None)
        kwargs["max_degree"] = min(kwargs["max_degree"], 4 if squared else 2)

    data = None
    for ij in get_batches(len(weights), args.batch_size_spin):
        grid = dict(weights=weights[ij], trafos=trafos[ij])
        if args.backend == "pairwise":
            # single grid points are not Hermitian
            values = compute_in_blocks(
                Partial(expval, **grid), X, n_results=2,
                batch_size=args.batch_size_proj, arrays=arrays, hermitian=False,
            )
        else:
            pair_batch = kwargs["pair_batch_size"]
            if pair_batch is None:
                pair_batch = tuned_pair_batch(
                    args, lambda candidate: expval.lower(X, X, arrays=arrays, **grid, **dict(kwargs, pair_batch_size=candidate)), len(ij) * X.ndet,
                    "postprocess projected H^2" if squared else "postprocess projection", 1 if squared else 128,
                    (X, arrays, grid), ghf_operator, kwargs["cached_batch_size"], kwargs["max_degree"], kwargs["factorization"],
                )
            values = expval(X, X, arrays=arrays, **grid, **dict(kwargs, pair_batch_size=pair_batch), progress=progress)
        values = [value.reshape(X.ndet, X.ndet) for value in values]
        data = values if data is None else [total + value for total, value in zip(data, values)]
    return tuple(data) if (arrays is not None or squared) else data[0]


def add_projection_args(parser):
    group = parser.add_argument_group("spin and point-group projection")
    group.add_argument("--project_spin", type=int, nargs="+", metavar="S", help="compute the spin, point-group and (for linear molecules) Lz projectors for these S values")
    group.add_argument("--batch_size_proj", type=positive_int, required=False, help="batch size of the projectors for --backend pairwise")
    group.add_argument("--batch_size_spin", type=positive_int, required=False, help="number of spin quadrature points computed at once (default: all)")
    group.add_argument("--projected_energy", action="store_true", help="also save ⟨H P⟩, for the energies of the projected states")
    group.add_argument("--projected_variance", action="store_true", help="also save ⟨H² P⟩, for their variances; implies --projected_energy, expensive")


def projection_setup(ops_dict, is_linear, mol, mo_coeff, S_projs):
    # the Lz grid is only needed for linear molecules
    trafos_lz = lz_rot_trafos(mol, mo_coeff, phis=None)[0] if is_linear else []
    print('available symbols:', list(ops_dict))
    if len(trafos_lz):
        print('available indices for lzphi weights: 0...', len(trafos_lz) - 1)
    else:
        print('Nonlinear molecule: using spin and point-group operations without an Lz grid.')
    lz_indices = range(len(trafos_lz)) if is_linear else [None]
    jobs = [(S_proj, symbol, i) for S_proj in S_projs for symbol in ops_dict for i in lz_indices]
    print(f"computing {len(jobs)} symmetry-projection grid points "
          f"({len(S_projs)} spin sectors x {len(ops_dict)} symbols x {len(lz_indices)} angular factors)")
    return trafos_lz, jobs


def projection_grid(X, args, jobs, *, xyzstr, basis, cart, mol, mo_coeff, ops_dict, trafos_lz, hamiltonian=None, hamiltonian_ghf=None):
    # yields (key, overlap, energy, squared) for every grid point; energy and squared are None unless requested
    ghf = not isinstance(X.nelec, tuple)
    wants_hamiltonian = args.projected_energy or args.projected_variance
    show_progress = progress_wanted(args)
    with tqdm(total=len(jobs), desc="projection grid", unit="point", disable=not show_progress) as bar:
        for S_proj, symbol, lzphii in jobs:
            symbol_key, trafos, weights = prep_symm_trafos(
                xyzstr, basis, mo_coeff, symbol, S_proj, None, lzphii=lzphii,
                ghf=ghf, cart=cart, ops_dict=ops_dict, mol=mol,
                trafos_lz=trafos_lz, verbose=False,
            )
            bar.set_postfix_str(symbol_key, refresh=True)
            nchunks = -(-len(weights) // (args.batch_size_spin or len(weights)))
            with tqdm(total=nchunks * -(-X.ndet // args.hybrid_cached_batch_size), desc=symbol_key, unit="bra", leave=False, disable=not (show_progress and args.backend != "pairwise")) as inner:
                _bar[0] = inner
                projected = compute_projected_overlap(
                    X, args, weights, trafos, ghf_operator=S_proj is not None,
                    arrays=(hamiltonian_ghf if (S_proj is not None and not ghf) else hamiltonian) if wants_hamiltonian else None,
                    squared=args.projected_variance, progress=_tick if show_progress and args.backend != "pairwise" else None,
                )
            if args.projected_variance:
                overlap, squared, energy = projected
            elif args.projected_energy:
                (overlap, energy), squared = projected, None
            else:
                overlap, energy, squared = projected, None, None
            del projected
            yield symbol_key, overlap, energy, squared
            bar.update(1)
