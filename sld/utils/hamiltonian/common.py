import jax
import jax.numpy as jnp
import numpy as np
import sparse

import jax.experimental.multihost_utils

from .hamiltonian import particle_hole_trafo
from .orb_conversion import spinorb_from_spatial2, to_desc_order

from si_prefix import si_format


def orbitals_pyscf(xyz_or_xyzstr, basis, spin=0, mo_coeff=None, unit="Angstrom", compute=True, sparse_=False, cutoff=1e-13, run_hf=True, cart=False, symmetry=True):
    from pyscf import gto, scf

    if ".xyz" == xyz_or_xyzstr[-4:]:
        # filename
        with open(xyz_or_xyzstr, "r") as f:
            xyzstr = f.read()
    else:
        # string
        xyzstr = xyz_or_xyzstr
    # if unit == Angstrom this converts it to AU
    atom = gto.mole.format_atom(gto.mole.fromstring(xyzstr, format="xyz"), unit=unit)
    mol = gto.M(atom=atom, basis=basis, spin=spin, unit="AU", cart=cart, symmetry=symmetry)

    nelec = tuple(map(int, mol.nelec))
    nao = int(mol.nao)

    if not compute:
        return nao, nelec, mo_coeff, None, None, None

    print("computing orbitals with pyscf")
    if True:
        h1e_ao = np.asarray(mol.intor("int1e_kin") + mol.intor("int1e_nuc"))
    h1e_ao = h1e_ao * (np.abs(h1e_ao) > cutoff)

    if sparse_:
        raise NotImplementedError
        # compute compressed version directly
        # h2e_ao = ...
    else:
        h2e_ao = np.asarray(mol.intor("int2e"))
        h2e_ao = h2e_ao * (np.abs(h2e_ao) > cutoff)
    enuc = np.asarray(float(mol.energy_nuc()))

    origin = np.zeros(3)
    with mol.with_common_orig(origin):
        dip_ao = np.asarray(mol.intor_symmetric("int1e_r", comp=3))
        angmom_ao = -1.0j * np.asarray(mol.intor("int1e_cg_irxp", comp=3))

    if mo_coeff is None:
        if run_hf:
            print("running hartree-fock")
            mf = scf.RHF(mol)
            # mf.verbose = 0
            mf.kernel()
            # mf.analyze()
            mo_coeff = np.asarray(mf.mo_coeff)
            if jax.process_count() > 1:
                mo_coeff = jax.experimental.multihost_utils.broadcast_one_to_all(mo_coeff)
        else:
            print("diagonalizing overlap matrix")
            ovlp = mol.intor("int1e_ovlp")
            s, U = np.linalg.eigh(ovlp)
            mo_coeff = np.einsum("ik,k,jk->ij", U, 1.0 / np.sqrt(s), U.conj())

    return nao, nelec, mo_coeff, enuc, h1e_ao, h2e_ao, dip_ao, angmom_ao, mol


def prepare_hamiltonian(enuc, h1e_mo, h2e_mo, ghf, particle_hole=False, sparse_=False, precomp_t=True):
    # if sparse_ == True, h2e_mo needs to be the 8-fold compressed one

    if particle_hole:
        if sparse_:
            raise NotImplementedError
        if ghf:
            raise NotImplementedError  # TODO check it's the same
        enuc, h1e_mo, h2e_mo = particle_hole_trafo(enuc, h1e_mo, 0.5 * h2e_mo)
        h2e_mo = 2 * h2e_mo
    if ghf:
        if sparse_:
            raise NotImplementedError
        # double number of orbitals
        # TODO sparse/low-rank version for h2eg
        h1eg, h2eg = spinorb_from_spatial2(h1e_mo, 0.5 * h2e_mo.transpose(0, 2, 3, 1))
        h2eg = to_desc_order(h2eg, 1e-12)
        hamiltonian = h2eg, h1eg, enuc
    else:
        if sparse_:
            raise NotImplementedError
        else:
            # convert to jax if not already
            enuc = jnp.asarray(enuc)
            h1e_mo = jnp.asarray(h1e_mo)
            h2e_mo = jnp.asarray(h2e_mo)
            hamiltonian = 0.5 * h2e_mo.transpose(0, 2, 3, 1), h1e_mo, enuc, h2e_mo
            # pre-compute once here
            if precomp_t:
                H2e, *rest = hamiltonian
                Ht = H2e - H2e.transpose(0, 1, 3, 2)
                Htt = Ht - Ht.transpose(1, 0, 2, 3)
                hamiltonian = (H2e, Ht, Htt), *rest

    return hamiltonian
