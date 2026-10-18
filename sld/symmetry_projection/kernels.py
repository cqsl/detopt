import jax
import jax.numpy as jnp
import numpy as np
import math
import scipy.linalg

from pyscf.symm.Dmatrix import dmatrix  # little-d only; see total_spin_trafo

from functools import partial

from sld.mels._src.pairwise.derivatives import hess
from sld.mels._src.pairwise.derivatives_uhf import hess_spin
from sld.utils.hamiltonian.orb_conversion import uhf2ghf
from .quadrature import quad_points_cosbeta, quad_points_uniform

def get_mo_trafo(trafo, mo_coeff, mo_coeff_symm, cutoff=1e-15):
    # trafo: orbital transformation
    # mo_coeff: basis of the determinants
    # mo_coeff_symm: basis of the transformation; pass identity matrix for AO basis
    trafo = np.array(trafo)
    mo_coeff = np.array(mo_coeff)
    mo_coeff_symm = np.array(mo_coeff_symm)
    if not np.allclose(mo_coeff, mo_coeff_symm):
        # transform from the basis of the operation to the basis of the determinants
        if trafo.ndim == 1:
            trafo = np.diag(trafo)
        trafo = (mo_coeff.T @ np.linalg.inv(mo_coeff_symm.T) @ trafo @ mo_coeff_symm.T @ np.linalg.inv(mo_coeff.T)).T
    trafo = trafo * (np.abs(trafo)>cutoff)
    return trafo

def Ry(no, beta):
    # e^{- i \beta Sy}
    I = jnp.eye(no)
    c = jnp.cos(0.5*beta)
    s = jnp.sin(0.5*beta)
    return jnp.block([[c * I, s * I], [-s * I, c * I]])

def Rz(no, alpha):
    # e^{- i \alpha Sz}
    I = jnp.eye(no)
    em = jnp.exp(-0.5 * alpha * 1.j)
    ep = jnp.exp(+0.5 * alpha * 1.j)
    return jnp.block([[em * I, 0*I], [0*I, ep * I]])

def R(no, alpha, beta, gamma):
    # e^{- i \alpha Sz} e^{- i \beta Sy} e^{- i \gamma Sz}
    return Rz(no, alpha)@Ry(no, beta)@Rz(no, gamma)


def apply_spin_rotation(X, alpha, beta, gamma, transpose=False):
    no = X.shape[-1] // 2
    up, dn = X[..., :no], X[..., no:]
    am, ap = jnp.exp(-0.5j * alpha), jnp.exp(0.5j * alpha)
    gm, gp = jnp.exp(-0.5j * gamma), jnp.exp(0.5j * gamma)
    c, s = jnp.cos(0.5 * beta), jnp.sin(0.5 * beta)
    if transpose:
        out_up = am * (c * gm * up + s * gp * dn)
        out_dn = ap * (-s * gm * up + c * gp * dn)
    else:
        out_up = gm * (am * c * up - ap * s * dn)
        out_dn = gp * (am * s * up + ap * c * dn)
    return jnp.concatenate([out_up, out_dn], axis=-1)


def apply_spatial_trafo(X, t):
    no = X.shape[-1] // 2
    return jnp.concatenate([X[..., :no] @ t, X[..., no:] @ t], axis=-1)


def apply_trafo(X, angles, spatial=None, transpose=False):
    if spatial is not None:
        X = apply_spatial_trafo(X, spatial.T if transpose else spatial)
    return apply_spin_rotation(X, angles[0], angles[1], angles[2], transpose=transpose)


def _grid_apply(X, angles, spatial, transpose=False):
    per_spin = lambda Y: jax.vmap(lambda ang: apply_trafo(Y, ang, transpose=transpose))(angles)
    if spatial is None:
        return per_spin(X)
    Xs = jax.vmap(lambda t: apply_spatial_trafo(X, t.T if transpose else t))(spatial)
    out = jax.vmap(per_spin)(Xs)
    return out.reshape(-1, *out.shape[2:])


def _resolve_trafos(trafos, angles, spatial):
    if (trafos is None) == (angles is None):
        raise ValueError('pass exactly one of trafos= (dense) or angles= (factored)')
    if trafos is not None and spatial is not None:
        raise ValueError('spatial= is only meaningful together with angles=')
    return trafos, angles, spatial


def total_spin_trafo(no, nelec, S, M=None, Mprime=None, uhf=True, deg1=None, deg2=None, Imax=None, materialize=True):
    #project onto |M, S><M', S|
    # uhf: state is eigenstate of M
    if isinstance(nelec, tuple):
        # for the bound on the number of quad points
        ne = sum(nelec)
    else:
        ne = nelec

    if Mprime is None:
        Mprime = M

    if S is None:
        raise NotImplementedError('please sum by hand over all S projectors')

    if uhf:
        # we have eigenstates of Sz with fixed M
        # S and M are true quantum numbers, not 2S/2M: Sz = (n_up - n_dn)/2
        if M is not None:
            assert Mprime == M
            assert nelec[0]-nelec[1] == 2*M, f'{M=} must equal (n_up - n_dn)/2 = {(nelec[0]-nelec[1])/2}'
        else:
            assert Mprime is None
            M = Mprime = (nelec[0]-nelec[1]) / 2
    else: # ghf
        if M is None:
            raise NotImplementedError('please sum by hand over all M projectors or project onto specific M')

    # pyscf's (d/D)matrix only implement integer angular momenta
    if (2*S) % 2 != 0:
        raise NotImplementedError(
            f'half-integer S={S} (odd electron number) is not supported: '
            'pyscf.symm.Dmatrix only implements integer angular momenta'
        )
    for name, val in (('M', M), ("M'", Mprime)):
        if float(S + val) != int(S + val) or abs(val) > S:
            raise ValueError(f'{name}={val} is not a valid magnetic quantum number for S={S}')
    iM, iMprime = int(S + M), int(S + Mprime)
    # exact quadrature degrees (Shimizu & Tsunoda, DOI: 10.1016/j.cpc.2022.108583): with Imax = min(ne, 2*no - ne)/2,
    # beta needs ceil((S + Imax + 1)/2) Gauss-Legendre points in cos(beta), alpha/gamma Imax + max(|M|, |M'|) + 1 uniform ones
    if Imax is None:
        Imax = min(ne, 2*no - ne) / 2
    if deg1 is None:
        deg1 = max(math.ceil((S + Imax + 1) / 2), 1)
    xs1, w1 = quad_points_cosbeta(deg1)  # w1 already contains the sin(beta) measure


    if uhf:
        # we have eigenstates of Sz with fixed M
        angles = np.stack([np.zeros_like(xs1), xs1, np.zeros_like(xs1)], axis=-1)
        trafos = jax.vmap(partial(Ry, no))(xs1) if materialize else jnp.asarray(angles)
        # the dmatrix rows and columns are for M \in [-S, ..., +S]
        # so e.g. the M=0 is at index S
        assert M == Mprime
        wigner_d_weight = np.array([dmatrix(S, b)[iM,iMprime] for b in xs1], dtype=float)
        weights = (2*S+1)/2 * wigner_d_weight * w1
    else:
        if deg2 is None:
            deg2 = math.ceil(Imax + max(abs(M), abs(Mprime))) + 1
        xs2, w2 = quad_points_uniform(deg2)
        a, b, g = np.meshgrid(xs2, xs1, xs2, indexing="ij")
        # D^S_MK(a,b,g) = exp(i M a) d^S_MK(b) exp(i K g)
        # can't use pyscf's Dmatrix here since it is real-basis (The small-d dmatrix(S,beta) is complex-basis and fine.)
        d_beta = np.array([dmatrix(S, bb)[iM, iMprime] for bb in xs1], dtype=float)
        wigner_D_weight = (np.exp(1j * M * xs2)[:, None, None]
                           * d_beta[None, :, None]
                           * np.exp(1j * Mprime * xs2)[None, None, :])
        wa, wb, wg = np.meshgrid(w2, w1, w2, indexing="ij")
        weights = (2*S+1)/np.pi/np.pi/8 * (wa*wb*wg*wigner_D_weight).ravel()
        if materialize:
            trafos = jax.vmap(partial(R, no))(a.ravel(), b.ravel(), g.ravel())
        else:
            trafos = jnp.asarray(np.stack([a.ravel(), b.ravel(), g.ravel()], axis=-1))
    return weights, trafos

def lz_rot_trafos(mol, mo_coeff, phis=None):
    # exp(-i phi Lz)

    # assumes the principal axis is the z axis

    origin = np.zeros(3)
    with mol.with_common_orig(origin):
        dip_ao = np.asarray(mol.intor_symmetric("int1e_r", comp=3))
        angmom_ao = -1.0j * np.asarray(mol.intor("int1e_cg_irxp", comp=3))

    Lz_ao = angmom_ao[2]
    S_ao = mol.intor('int1e_ovlp')

    # find orthogonal basis in which Lz is diagonal
    eigvals, mo_coeff_symm = scipy.linalg.eigh(Lz_ao, S_ao)
    eigvals = np.round(eigvals).astype(int)
    lmax = np.abs(eigvals).max()

    if phis is None:
        N = 2 * sum(mol.nelec) * lmax + 1
        phis = 2*np.pi*np.arange(N)/N
    else:
        N = len(phis)
    trafos_lz = [get_mo_trafo(np.diag(np.exp(-1.j * phi * eigvals)), mo_coeff, mo_coeff_symm) for phi in phis]

    # weights for each value of Lz, and for |Lz| (sum of the +L and -L projectors)

    lm = lmax*sum(mol.nelec)
    symbols_lz = list(range(-lm-1, lm + 2))
    weights_lz = {l: np.exp(-1j * l * phis)/N for l in symbols_lz}

    # sum projector of +L and -L
    symbols_l_abs = list('ΣΠΔΦ')[:lmax+1]
    weights_l_abs = {s: (2 if l>0 else 1) * np.cos(l * phis)/N for l, s in enumerate(symbols_l_abs)}

    return trafos_lz, phis, weights_lz, symbols_lz, weights_l_abs, symbols_l_abs


def expval_proj_ghf(expval_impl, Uconj, V, deriv_indices=None, hamiltonian=None, weights=None, trafos=None, angles=None, spatial=None):
    assert weights is not None
    trafos, angles, spatial = _resolve_trafos(trafos, angles, spatial)
    # compute <Psi | H P | Psi> and <Psi| P |Psi>
    if trafos is not None:
        V_p = jnp.einsum("ij,ajk->aik", V, trafos)
    else:
        V_p = _grid_apply(V, angles, spatial)

    oo, ee = jax.vmap(lambda v: expval_impl(Uconj, v, hamiltonian=hamiltonian))(V_p)
    oo = jnp.einsum('a,a...->...', weights, oo)
    ee = jnp.einsum('a,a...->...', weights, ee)
    return oo, ee


def expval_proj_uhf_via_ghf(expval_impl, Uconju, Uconjd, Vu, Vd, deriv_indices=None, hamiltonian=None, weights=None, trafos=None, angles=None, spatial=None):
    # uhf determinants, but a ghf operator which mixes spin up and down
    assert weights is not None
    Uconj = uhf2ghf(Uconju, Uconjd)
    V = uhf2ghf(Vu, Vd)
    return expval_proj_ghf(expval_impl, Uconj, V, deriv_indices=None, hamiltonian=hamiltonian, weights=weights, trafos=trafos, angles=angles, spatial=spatial)


def expval_proj_uhf(expval, Uconju, Uconjd, Vu, Vd, deriv_indices=None, hamiltonian=None, weights=None, trafos=None, angles=None, spatial=None):
    assert weights is not None
    assert trafos is not None
    assert angles is None and spatial is None  # factored grids not implemented here, use dense trafos

    Vup = jnp.einsum('aij,xj->axi', trafos, Vu)
    Vdp = jnp.einsum('aij,xj->axi', trafos, Vd)
    f = partial(expval, Uconju, Uconjd, deriv_indices=deriv_indices, hamiltonian=hamiltonian)
    S, H = jax.lax.map(lambda x: f(*x), (Vup, Vdp), batch_size=0)
    S = jnp.einsum("a...,a->...", S, weights)
    H = jnp.einsum("a...,a->...", H, weights)

    return S, H


def hess_proj_ghf(impl, Uconj, V, deriv_indices=None, hamiltonian=None, weights=None, trafos=None):
    # ⟨∂U|Ô P̂|∂V⟩ = Σ_g w_g ⟨∂U|Ô|∂(V R_g)⟩, the ket derivative contracted with R_gᵀ
    V_p = jnp.einsum("ij,ajk->aik", V, trafos)
    S, H = jax.lax.map(partial(hess, impl, Uconj, deriv_indices=deriv_indices, hamiltonian=hamiltonian), V_p)
    S = jnp.einsum("aij,akj,a->ik", S, trafos, weights)
    H = jnp.einsum("aij,akj,a->ik", H, trafos, weights)
    return S, H


def hess_proj_uhf_via_ghf(impl, Uconju, Uconjd, Vu, Vd, deriv_indices=None, hamiltonian=None, weights=None, trafos=None):
    # uhf determinants with a spin rotating projector, the hamiltonian and trafos are GHF
    s, t, (ku, kd), (lu, ld) = deriv_indices
    ne_u = Uconju.shape[-2]
    deriv_indices_ghf = (1 - s) * ku + s * (ne_u + kd), (1 - t) * lu + t * (ne_u + ld)
    S, H = hess_proj_ghf(impl, uhf2ghf(Uconju, Uconjd), uhf2ghf(Vu, Vd), deriv_indices=deriv_indices_ghf, hamiltonian=hamiltonian, weights=weights, trafos=trafos)
    # the spin block of the derivative orbitals
    nao = Uconju.shape[-1]
    def block(A):
        A = A.reshape(2, nao, 2 * nao).reshape(2, nao, 2, nao)
        return jax.lax.select(s == 0, jax.lax.select(t == 0, A[0, :, 0], A[0, :, 1]), jax.lax.select(t == 0, A[1, :, 0], A[1, :, 1]))
    return block(S), block(H)


def hess_proj_uhf(impl, Uconju, Uconjd, Vu, Vd, deriv_indices=None, hamiltonian=None, weights=None, trafos=None):
    # spatial projector, the same trafo for up and down
    Vup = jnp.einsum('aij,xj->axi', trafos, Vu)
    Vdp = jnp.einsum('aij,xj->axi', trafos, Vd)
    f = partial(hess_spin, impl, Uconju, Uconjd, deriv_indices=deriv_indices, hamiltonian=hamiltonian)
    S, H = jax.lax.map(lambda x: f(*x), (Vup, Vdp))
    S = jnp.einsum("aij,ajk,a->ik", S, trafos, weights)
    H = jnp.einsum("aij,ajk,a->ik", H, trafos, weights)
    return S, H
