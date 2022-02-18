from collections import Counter

import rdkit

# These filters should return 'False' if it passes the criteria


def h2(mol):
    """no atom shared by two small rings"""
    return mol.HasSubstructMatch(
        rdkit.Chem.MolFromSmarts("[R2r3,r4]([R1r3,r4])([R1r3,r4])[R1r3,r4]")
    )


def h3(mol):
    """no bridgehead in 3 rings"""
    return mol.HasSubstructMatch(rdkit.Chem.MolFromSmarts("[R3]"))


def h4(mol):
    """one small ring"""
    ring_counts = Counter((len(ring) for ring in mol.GetRingInfo().AtomRings()))
    return (ring_counts[3] + ring_counts[4]) > 1


def s1(mol):
    """no allenes"""
    return mol.HasSubstructMatch(rdkit.Chem.MolFromSmarts("C=C=C"))


def s2(mol):
    """no unsaturations in 3-membered rings"""
    for bonds in filter(lambda x: len(x) == 3, mol.GetRingInfo().BondRings()):
        for bond in bonds:
            if mol.GetBondWithIdx(bond).GetBondTypeAsDouble() > 1:
                return True

    return False


def s3(mol):
    """at most one sp2-center in 4-membered rings"""
    for atoms in filter(lambda x: len(x) == 4, mol.GetRingInfo().AtomRings()):
        c_atoms = filter(lambda x: mol.GetAtomWithIdx(x).GetSymbol() == "C", atoms)
        atom_centers = Counter(
            (mol.GetAtomWithIdx(atom).GetHybridization() for atom in c_atoms)
        )
        if atom_centers[rdkit.Chem.rdchem.HybridizationType.SP2] > 1:
            return True

    return False


def s4(mol):
    """No triple bonds in ring"""
    return mol.HasSubstructMatch(rdkit.Chem.MolFromSmarts("[R]#[R]"))


def f2(mol):
    """at most one sp2-center in 4-membered rings"""
    for atoms in filter(lambda x: len(x) <= 4, mol.GetRingInfo().AtomRings()):
        atom_types = Counter((mol.GetAtomWithIdx(atom).GetSymbol() for atom in atoms))
        if atom_types["O"] + atom_types["S"] + atom_types["N"] > 1:
            return True

    return False


def check_all_filters(mol):
    """Returns True if the molecule passes, else false."""
    for fn in [h2, h3, h4, s1, s2, s3, s4, f2]:
        if fn(mol):
            return False

    return True
