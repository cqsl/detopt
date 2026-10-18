# Patch MolSym so that the irreps follow the usual (Mulliken) convention: the principal axis of linear molecules is the molecular axis,
# and the C2 axes of pairs of symmetry-equivalent atoms are considered, so that e.g. z is the C=C axis of ethylene (pi is b3u, the pi -> pi* state B1u).
# For MolSym 178e029.

import numpy as np

import molsym
import molsym.pgdetect.flowchart_helper as _helper
import molsym.pgdetect.flowchart as _flowchart
import molsym.symtext.symtext as _symtext
from molsym.symtools import calcmoit

_find_point_group = _flowchart.find_point_group
_find_rotation_sets = _helper.find_rotation_sets


def find_point_group(mol):
    pg, (paxis, saxis) = _find_point_group(mol)
    evals, evecs = np.linalg.eigh(calcmoit(mol))
    if np.isclose(evals[0], 0.0, atol=mol.tol):
        paxis = evecs[:, 0]
    return pg, (paxis, saxis)


def find_rotation_sets(mol, SEAs):
    out = []
    for sea in SEAs:
        out += _find_rotation_sets(mol, [sea])
        if len(sea.subset) == 2:
            out.append([_helper.RotationElement(sea.axis, 2)])
    return out


_helper.find_rotation_sets = find_rotation_sets
_flowchart.find_rotation_sets = find_rotation_sets
_flowchart.find_point_group = find_point_group
_symtext.find_point_group = find_point_group
molsym.find_point_group = find_point_group
