import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag
from pyscf.symm import Dmatrix as dm
from pyscf import lib
from sld.utils.molecule import mol_from_xyz

_table2_pm = ('sigma_v',), {1: '+', -1:'-', 0:''}
_table2_ug = ('i',), {1: 'g', -1:'u'}
_table2_lambd = ('lz',), {0: "Σ", 1: "Π", 2: "Δ", 3: "Φ", -1: "Π", -2: "Δ", -3: "Φ"}
_table2_spin = ('s2',), {s*(s+1):str(2*s+1) for s in range(10)}


def _cart_rot_matrix(l, pp):
    if l == 0:
        return np.ones((1, 1))
    cidx = np.sort(lib.cartesian_prod([(0, 1, 2)] * l), axis=1)
    addr = 0
    affine = np.ones((1, 1))
    for i in range(l):
        nd = affine.shape[0] * 3
        affine = np.einsum('ik,jl->ijkl', affine, pp).reshape(nd, nd)
        addr = addr * 3 + cidx[:, i]
    uniq_addr, rev_addr = np.unique(addr, return_inverse=True)
    ncart = (l + 1) * (l + 2) // 2
    trans = np.zeros((ncart, ncart))
    for i, k in enumerate(rev_addr):
        trans[k] += affine[i, uniq_addr]
    return trans


def ao_rot_atom(mol, ia, R):
    det = np.linalg.det(R)
    if det < 0:
        Rproper = -R
        parity = -1
    else:
        Rproper = R
        parity = +1

    with warnings.catch_warnings():
        # gimbal lock is harmless, the angles still give the right rotation
        warnings.simplefilter('ignore', UserWarning)
        alpha, beta, gamma = Rotation.from_matrix(Rproper).as_euler('ZYZ')
    sh0, sh1, ao0, ao1 = mol.aoslice_by_atom()[ia]

    l_max = max(mol.bas_angular(ish) for ish in range(sh0, sh1))
    if mol.cart:
        pp = dm.Dmatrix(1, alpha, beta, gamma, reorder_p=True)
        Dcache = [np.ones((1, 1))]
        for l in range(1, l_max + 1):
            Dcache.append(_cart_rot_matrix(l, pp))
    else:
        Dcache = [dm.Dmatrix(l, alpha, beta, gamma, reorder_p=True) for l in range(l_max + 1)]

    blocks = []
    for ish in range(sh0, sh1):
        l = mol.bas_angular(ish)
        D = Dcache[l]
        if parity < 0:
            D = D * (-1)**l
        nctr = mol.bas_nctr(ish)
        blocks.append(np.kron(np.eye(nctr), D))
    return block_diag(*blocks)

def build_ao_transform(mol, R, perm):
    nao = mol.nao
    T = np.zeros((nao, nao))
    ao_slices = mol.aoslice_by_atom()

    for a in range(mol.natm):
        b = perm[a]
        sh0, sh1, p0, p1 = ao_slices[a]
        _,  _,  q0, q1 = ao_slices[b]

        R_a = ao_rot_atom(mol, a, R)
        T[q0:q1, p0:p1] = R_a

    return T.T

def prep_pg_symm(xyzstr, basis, linear_subgroup="dinftyh", cart=False):
    from sld.utils.molsym_patched import molsym
    import qcelemental
    msmol = molsym.Molecule.from_schema(qcelemental.models.v2.Molecule.from_data(xyzstr).model_dump())
    st = molsym.Symtext.from_molecule(msmol)
    print('Symmetry group used:', st.pg)
    pyscfmol = mol_from_xyz(xyzstr, basis, cart=cart)
    is_linear, reverse_rotate = st.pg.is_linear, st.reverse_rotate
    if is_linear and linear_subgroup.lower() in {"d2h", "d4h"}:
        # a finite subgroup of D∞h, then as for the other point groups
        st = st.subgroup_symtext(linear_subgroup.capitalize())
    if st.pg.is_linear: # Coov and Dooh
        if linear_subgroup.lower() not in {"dinftyh", "default"}:
            raise ValueError(f"unsupported linear subgroup {linear_subgroup}")
        ops_dict = {}
        ops_dict['E'] = build_ao_transform(pyscfmol, np.eye(3), st.atom_map[:,0])
        R_sigma = np.diag([1,-1,1])
        ops_dict['sigma_v'] = build_ao_transform(pyscfmol, st.reverse_rotate@R_sigma@st.rotate_to_std, st.atom_map[:,0])
        symbols = ['E', 'sigma_v']
        if st.pg.str in {'D0h', 'Dinfh'}:
            R_inv = np.diag([-1,-1,-1])
            ops_dict['i'] = build_ao_transform(pyscfmol, st.reverse_rotate@R_inv@st.rotate_to_std, st.atom_map[:,1])
            ops_dict['C2'] = build_ao_transform(pyscfmol, st.reverse_rotate@R_sigma@R_inv@st.rotate_to_std, st.atom_map[:,1])
            symbols.append('i')
            symbols.append('C2')
            table2 = [_table2_spin, _table2_lambd, _table2_pm, _table2_ug]
        else:
            table2 = [_table2_spin, _table2_lambd, _table2_pm]
    else:

        irreps = [x.symbol for x in  st.irreps]
        symbols = [x.symbol for x in st.symels]
        rotation_matrices = [st.reverse_rotate@x.rrep@st.rotate_to_std for x in st.symels]
        atom_perm = st.atom_map.T
        irrep_mats = st.irrep_mats

        ops_dict = {}
        for i, s in enumerate(symbols):
            ops_dict[s] = build_ao_transform(pyscfmol, rotation_matrices[i], atom_perm[i])

        weights = np.array([np.trace(irrep_mats[i], axis1=1, axis2=2)/irrep_mats[i].shape[0]*irrep_mats[i].shape[1] for i in irreps]) # irreps x symbols

        if st.pg.str in ['D2', 'D2h']:
            irreps = [ir.replace('A_1', 'A') for ir in irreps]
        elif st.pg.str == 'Cs':
            irreps = [ir.replace("A''", 'A"') for ir in irreps]

        t2 = (irreps, symbols, weights)
        table2 = [_table2_spin, (symbols, t2)]


    return ops_dict, table2, reverse_rotate, is_linear


def table_to_h5(group, symbols, table2, is_linear):
    # the symmetry table, so that reading the postprocessed file does not need molsym
    if table2 is None:
        return {}
    data = {"symm_group": group, "symm_linear": is_linear, "symm_symbols": list(symbols)}
    if isinstance(table2[1][1], tuple):  # character table, also for linear molecules with --linear_subgroup
        irreps, _, characters = table2[1][1]
        data.update(symm_irreps=list(irreps), symm_characters=np.asarray(characters))
    return data


def symmetry_table(data, xyzstr, basis, cart=False):
    # table2 and is_linear stored by table_to_h5, recomputed with molsym for files without them
    from sld.utils.io import to_str
    if "symm_group" not in data:
        _, table2, _, is_linear = prep_pg_symm(xyzstr, basis, cart=cart)
        return table2, is_linear
    print('Symmetry group used:', to_str(data["symm_group"]))
    symbols = to_str(data["symm_symbols"])
    if "symm_irreps" in data:
        table2 = [_table2_spin, (symbols, (to_str(data["symm_irreps"]), symbols, np.asarray(data["symm_characters"])))]
    else:
        table2 = [_table2_spin, _table2_lambd, _table2_pm] + ([_table2_ug] if "i" in symbols else [])
    return table2, bool(data["symm_linear"])


def guess_states_proj(postprocess_data, table2neo, tol=0.1, key='_full_degenerate_diag'):
    irreps, symbols, weights = table2neo

    measurement_data = np.array([postprocess_data[k+key] for k in symbols]) # symbols x states
    # ⟨Ψ_k|P_Γ|Ψ_k⟩ / ⟨Ψ_k|Ψ_k⟩
    proj_coeff = np.einsum('sx,is->xi', measurement_data, weights).real # states x irreps

    res = []
    for pi in proj_coeff:
        j = np.argmax(pi)
        if pi[j] > 1-tol:
            res.append(irreps[j].replace('_', ''))
        else:
            res.append('?')
    return res, irreps, symbols, proj_coeff


def prep_symm_trafos(
    xyzstr, basis, mo_coeff, symbol=None, S_proj=None, M_proj=None, lzphii=None,
    ghf=False, cart=False, linear_subgroup="dinftyh", *, ops_dict=None, mol=None,
    trafos_lz=None, verbose=True,
):
    from .kernels import get_mo_trafo, lz_rot_trafos, total_spin_trafo

    # version for the postprocessing, computes all values of Lz
    if ops_dict is None:
        ops_dict, *_ = prep_pg_symm(
            xyzstr, basis, linear_subgroup=linear_subgroup, cart=cart,
        )
    if mol is None:
        mol = mol_from_xyz(xyzstr, basis, cart=cart)
    nao = mol.nao
    if trafos_lz is None:
        trafos_lz, *_ = lz_rot_trafos(mol, mo_coeff, phis=None)
    if verbose:
        print('available symbols:', list(ops_dict.keys()))
        print('available indices for lzphi weights: 0...', len(trafos_lz) - 1)

    if symbol is not None:
        # ops_dict holds AO-basis transforms
        t_pg = get_mo_trafo(ops_dict[symbol], mo_coeff, np.eye(nao))
        symbol_key = symbol
    else:
        symbol_key = ''
        t_pg = np.eye(nao)
    if lzphii is not None:
        lz_trafo = trafos_lz[lzphii]
        t_pg = t_pg@lz_trafo
        symbol_key = f'PLz{lzphii}_{symbol_key}'

    t_pg_ghf = np.block([[t_pg, t_pg*0],[t_pg*0, t_pg]])
    if S_proj is not None:
        if ghf and M_proj is None:
            # GHF is no eigenstate of Sz, P_S = sum_M P_S,MM
            parts = [total_spin_trafo(nao, mol.nelec, S_proj, M=M, uhf=False) for M in np.arange(-S_proj, S_proj + 1)]
            w_S2, t_S2 = np.concatenate([w for w, _ in parts]), np.concatenate([np.asarray(t) for _, t in parts])
        else:
            w_S2, t_S2 = total_spin_trafo(nao, mol.nelec, S_proj, M=M_proj, uhf=not ghf)
        t_comb = np.einsum('aij,jk->aik', np.asarray(t_S2), t_pg_ghf)
        weights=w_S2
        trafos=t_comb
        symbol_key = f'PS{S_proj}_{symbol_key}'
    else:
        weights=np.ones(1)
        trafos=(t_pg_ghf if ghf else t_pg)[None]

    if verbose:
        print('computing', symbol_key)
        print('number of weights', len(weights))
    return symbol_key, trafos, weights


def prep_vap_trafos(xyzstr, basis, mo_coeff, nelec, S_proj=None, irrep=None, lz_proj=None, cart=False, linear_subgroup="dinftyh", charge=0):
    # the grid of P̂ = P̂_S P̂_Γ P̂_|Λ| for sld-optimize, (weights, trafos) with GHF trafos if ghf or S_proj
    from .kernels import get_mo_trafo, lz_rot_trafos, total_spin_trafo

    ghf = not isinstance(nelec, tuple)
    mol = mol_from_xyz(xyzstr, basis, cart=cart, charge=charge)
    nao = mol.nao
    grids = []
    if irrep is not None or lz_proj is not None:
        ops_dict, table2, _, is_linear = prep_pg_symm(xyzstr, basis, linear_subgroup=linear_subgroup, cart=cart)
    # linear molecules: parities of D∞h, or the irreps of the finite subgroup d2h/d4h
    if irrep is not None and is_linear and linear_subgroup == "dinftyh":
        # (Ê ± R̂)/2 for the parities, e.g. "g" or "u-"
        parities = {'g': ('i', 1), 'u': ('i', -1), '+': ('sigma_v', 1), '-': ('sigma_v', -1)}
        available = [p for p, (symbol, _) in parities.items() if symbol in ops_dict]
        print(f"available parities: {' '.join(available)} (or the irreps of --linear_subgroup d2h or d4h)")
        if not irrep or not set(irrep) <= set(available) or len({parities[p][0] for p in irrep}) != len(irrep):
            raise ValueError(f"for linear molecules --project_irrep takes the parities {available}, at most one of each pair, e.g. g or u-")
        for p in irrep:
            symbol, sign = parities[p]
            grids.append((np.array([0.5, 0.5 * sign]), np.stack([np.eye(nao), get_mo_trafo(ops_dict[symbol], mo_coeff, np.eye(nao))])))
    elif irrep is not None:
        symbols, (irreps, _, weights_pg) = table2[1]
        irreps = [x.replace('_', '') for x in irreps]
        print(f"available irreps{' of ' + linear_subgroup.capitalize() if is_linear else ''}:", " ".join(irreps))
        if irrep not in irreps:
            raise ValueError(f"unknown irrep {irrep}, available: {irreps}")
        grids.append((np.asarray(weights_pg[irreps.index(irrep)]), np.stack([get_mo_trafo(ops_dict[s], mo_coeff, np.eye(nao)) for s in symbols])))
    if lz_proj is not None:
        if not is_linear:
            raise ValueError("--project_lz needs a linear molecule")
        trafos_lz, phis, *_ = lz_rot_trafos(mol, mo_coeff)
        # P̂_|Λ| = (2 - δ_Λ0)/N Σ_φ cos(Λφ) exp(-iφ L̂z), exact for |Lz| < N/2
        if not 0 <= lz_proj < len(phis) / 2:
            raise ValueError(f"|Lz| = {lz_proj} is not available, at most {(len(phis) - 1) // 2}")
        grids.append(((2 if lz_proj else 1) * np.cos(lz_proj * phis) / len(phis), np.stack(trafos_lz)))

    # product grid of the spatial projectors, they commute
    weights, trafos = np.ones(1), np.eye(nao)[None]
    for w, t in grids:
        weights = np.multiply.outer(weights, w).ravel()
        trafos = np.einsum('aij,bjk->abik', trafos, t).reshape(-1, nao, nao)
    if ghf or S_proj is not None:
        trafos = np.einsum('st,aij->asitj', np.eye(2), trafos).reshape(-1, 2 * nao, 2 * nao)
    if S_proj is not None:
        if ghf:
            # P̂_S = Σ_M P̂_S,MM
            parts = [total_spin_trafo(nao, nelec, S_proj, M=M, uhf=False) for M in np.arange(-S_proj, S_proj + 1)]
            w_S, t_S = np.concatenate([w for w, _ in parts]), np.concatenate([np.asarray(t) for _, t in parts])
        else:
            w_S, t_S = total_spin_trafo(nao, nelec, S_proj, uhf=True)
            w_S, t_S = np.asarray(w_S), np.asarray(t_S)
        weights = np.multiply.outer(w_S, weights).ravel()
        trafos = np.einsum('aij,bjk->abik', t_S, trafos).reshape(-1, 2 * nao, 2 * nao)
    print('number of projection grid points', len(weights))
    return weights, trafos
