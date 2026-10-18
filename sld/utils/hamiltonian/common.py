import jax
import jax.numpy as jnp
import numpy as np

import jax.experimental.multihost_utils

from .hamiltonian import particle_hole_trafo
from .orb_conversion import spinorb_from_spatial2, spinorb_one_body
from .two_body import antisymmetrize_two_body
from ..molecule import mol_from_xyz


def orbitals_pyscf(xyz_or_xyzstr, basis, spin=0, mo_coeff=None, unit="Angstrom", run_hf=True, cart=False, symmetry=True, charge=0):
    from pyscf import scf

    if ".xyz" == xyz_or_xyzstr[-4:]:
        # filename
        with open(xyz_or_xyzstr, "r") as f:
            xyzstr = f.read()
    else:
        # string
        xyzstr = xyz_or_xyzstr

    mol = mol_from_xyz(xyzstr, basis, spin=spin, cart=cart, symmetry=symmetry, unit=unit, charge=charge)
    nelec = tuple(map(int, mol.nelec))
    nao = int(mol.nao)

    print("computing orbitals with pyscf")
    h1e_ao = np.asarray(mol.intor("int1e_kin") + mol.intor("int1e_nuc"))
    h2e_ao = np.asarray(mol.intor("int2e"))
    enuc = np.asarray(float(mol.energy_nuc()))

    origin = np.zeros(3)
    with mol.with_common_orig(origin):
        dip_ao = np.asarray(mol.intor_symmetric("int1e_r", comp=3))
        angmom_ao = -1.0j * np.asarray(mol.intor("int1e_cg_irxp", comp=3))

    if mo_coeff is None:
        if run_hf:
            print("running hartree-fock")
            mf = scf.RHF(mol)
            mf.kernel()
            mo_coeff = np.asarray(mf.mo_coeff)
            if jax.process_count() > 1:
                mo_coeff = jax.experimental.multihost_utils.broadcast_one_to_all(mo_coeff)
        else:
            print("diagonalizing overlap matrix")
            ovlp = mol.intor("int1e_ovlp")
            s, U = np.linalg.eigh(ovlp)
            mo_coeff = np.einsum("ik,k,jk->ij", U, 1.0 / np.sqrt(s), U.conj())

    return nao, nelec, mo_coeff, enuc, h1e_ao, h2e_ao, dip_ao, angmom_ao, mol


def prepare_hamiltonian(enuc, h1e_mo, h2e_mo, ghf, particle_hole=False, ghf_keepspin=False):

    if particle_hole:
        if ghf:
            raise NotImplementedError("particle-hole transformation for GHF")
        enuc, h1e_mo, h2e_mo = particle_hole_trafo(enuc, h1e_mo, 0.5 * h2e_mo)
        h2e_mo = 2 * h2e_mo
    if ghf:
        # double number of orbitals
        if ghf_keepspin:
            h2e_so = 0.5 * h2e_mo.transpose(0, 2, 3, 1)
            h1eg = spinorb_one_body(h1e_mo)
            h2eg = (antisymmetrize_two_body(h2e_so),
                    -(h2e_so.transpose(1, 0, 2, 3) + h2e_so.transpose(0, 1, 3, 2)))
        else:
            h1eg, h2eg = spinorb_from_spatial2(h1e_mo, 0.5 * h2e_mo.transpose(0, 2, 3, 1))
            h2eg = antisymmetrize_two_body(h2eg)
        hamiltonian = {'H2e': h2eg,'H1e': h1eg, 'Enuc': enuc}

    else:
        enuc = jnp.asarray(enuc)
        h1e_mo = jnp.asarray(h1e_mo)
        h2e_mo = jnp.asarray(h2e_mo)
        H2e = 0.5 * h2e_mo.transpose(0, 2, 3, 1)
        H2e = antisymmetrize_two_body(H2e)
        hamiltonian = {'H2e': H2e, 'H1e': h1e_mo, 'Enuc': enuc, 'w': h2e_mo}
    return hamiltonian
