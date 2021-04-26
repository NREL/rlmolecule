from typing import Optional, Sequence
import rdkit
from rdkit import Chem
from rdkit.Chem import Mol, MolToSmiles

from rlmolecule.molecule.molecule_state import MoleculeState
from rlmolecule.molecule.builder.builder import AddNewAtomsAndBonds


class StableRadMoleculeState(MoleculeState):
    """
    A State implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES string is also stored for
    efficient hashing.
    """
    def __init__(
        self,
        molecule: Mol,
        builder: any,
        force_terminal: bool = False,
        smiles: Optional[str] = None,
    ) -> None:
        super(StableRadMoleculeState, self).__init__(molecule, builder, force_terminal, smiles)

    def get_next_actions(self) -> Sequence['StableRadMoleculeState']:
        result = []
        if not self._forced_terminal:
            if self.num_atoms < self.builder.max_atoms:
                result.extend(
                    (StableRadMoleculeState(molecule, self.builder) for molecule in self.builder(self.molecule)))

            if self.num_atoms >= self.builder.min_atoms:
                result.extend((StableRadMoleculeState(radical, self.builder, force_terminal=True)
                               for radical in build_radicals(self.molecule)))

        return result


def build_radicals(starting_mol):
    """Build organic radicals. """

    generated_smiles = set()

    for i, atom in enumerate(starting_mol.GetAtoms()):
        if AddNewAtomsAndBonds._get_free_valence(atom) > 0:
            rw_mol = rdkit.Chem.RWMol(starting_mol)
            rw_mol.GetAtomWithIdx(i).SetNumRadicalElectrons(1)

            Chem.SanitizeMol(rw_mol)
            smiles = Chem.MolToSmiles(rw_mol)
            if smiles not in generated_smiles:
                # This makes sure the atom ordering is standardized
                yield Chem.MolFromSmiles(smiles)
                generated_smiles.add(smiles)
