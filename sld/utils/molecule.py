def mol_from_xyz(xyzstr, basis, spin=None, cart=False, symmetry=True, unit='Angstrom', charge=0):
    # spin None: the lowest spin of the electron count, for uses that only need the basis
    from pyscf import gto
    if xyzstr.split('\n')[0].strip(' ').isdigit():
        fmt = 'xyz'
    elif xyzstr.split('\n')[0].strip(' ').isalpha():
        fmt = 'zmat'
    else:
        fmt = 'raw'
    atom = gto.mole.format_atom(gto.mole.fromstring(xyzstr, format=fmt), unit=unit)
    if spin is None:
        spin = (sum(gto.charge(a[0]) for a in atom) - charge) % 2
    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, unit="AU", cart=cart, symmetry=symmetry)
    return mol
