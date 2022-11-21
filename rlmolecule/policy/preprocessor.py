import os
from typing import Optional

import rdkit.Chem

# from nfp.preprocessing.mol_preprocessor import MolPreprocessor
from nfp.preprocessing import MolPreprocessor
from nfp.preprocessing.features import get_ring_size


def atom_featurizer(atom: rdkit.Chem.Atom) -> str:
    """Return an string representing the atom type
    :param atom: the rdkit.Atom object
    :return: a string representation for embedding
    """

    return str(
        (
            atom.GetSymbol(),
            atom.GetNumRadicalElectrons(),
            atom.GetFormalCharge(),
            atom.GetChiralTag().name,
            atom.GetIsAromatic(),
            get_ring_size(atom, max_size=6),
            atom.GetDegree(),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
    )


def bond_featurizer(bond: rdkit.Chem.Bond, flipped: bool = False) -> str:
    """Return a string representation of the given bond

    :param bond: The rdkit bond object
    :param flipped: Whether the bond is considered in the forward or reverse direction
    :return: a string representation of the bond type
    """
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()))
        )
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(), bond.GetBeginAtom().GetSymbol()))
        )

    bstereo = bond.GetStereo().name
    btype = str(bond.GetBondType())
    ring = "R{}".format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ""

    return " ".join([atoms, btype, ring, bstereo]).strip()


def load_preprocessor(saved_preprocessor_file: Optional[str] = None) -> MolPreprocessor:
    """Load the MolPreprocessor object from either the default json file or a provided
    data file

    :param saved_preprocessor_file: directory of the saved nfp.Preprocessor json data
    :return: a MolPreprocessor instance for the molecule policy network
    """
    preprocessor = MolPreprocessor(
        atom_features=atom_featurizer,
        bond_features=bond_featurizer,
        output_dtype="int64",
    )

    if not saved_preprocessor_file:
        saved_preprocessor_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "preprocessor.json"
        )

    preprocessor.from_json(saved_preprocessor_file)
    return preprocessor
