
from typing import Optional, Sequence
import rdkit
from rdkit import Chem
from rdkit.Chem import Mol, MolToSmiles

from rlmolecule.molecule.molecule_state import MoleculeState
from rlmolecule.molecule.molecule_building import get_free_valence, build_molecules


class StableRadMoleculeState(MoleculeState):
    """
    A State implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES string is also stored for
    efficient hashing.
    """

    def __init__(self,
                 molecule: Mol,
                 config: any,
                 force_terminal: bool = False,
                 smiles: Optional[str] = None,
                 ) -> None:
        super(StableRadMoleculeState, self).__init__(molecule, config, force_terminal, smiles)

    def get_next_actions(self) -> Sequence['StableRadMoleculeState']:
        result = []
        if not self._forced_terminal:
            if self.num_atoms < self.config.max_atoms:
                # result.extend((StableRadMoleculeState(molecule, self.config) for molecule in
                #             build_molecules(
                #                 self.molecule,
                #                 atom_additions=self.config.atom_additions,
                #                 stereoisomers=self.config.stereoisomers,
                #                 sa_score_threshold=self.config.sa_score_threshold,
                #                 tryEmbedding=self.config.tryEmbedding
                #             )))
                for molecule in build_molecules(
                        self.molecule,
                        atom_additions=self.config.atom_additions,
                        stereoisomers=self.config.stereoisomers,
                        sa_score_threshold=self.config.sa_score_threshold,
                        tryEmbedding=self.config.tryEmbedding):
                    yield StableRadMoleculeState(molecule, self.config)

            if self.num_atoms >= self.config.min_atoms:
                # TODO is this the right approach?
                # Throw away the build_molecules() results once we've reached the radicals step
                # result = list(
                #     StableRadMoleculeState(radical, self.config, force_terminal=True) \
                #     for radical in build_radicals(self.molecule))
                for radical in build_radicals(self.molecule):
                    yield StableRadMoleculeState(radical, self.config, force_terminal=True)

        return result

    # Original code to get the next actions
    # def build_children(self):
    #     if self.terminal:
    #         raise RuntimeError("Attemping to get children of terminal node")

    #     if self.GetNumAtoms() < config.max_atoms:
    #         for mol in build_molecules(self, **config.build_kwargs):
    #             if self.G.has_node(mol):
    #                 # Check if the graph already has the current mol
    #                 yield self.G.nodes[mol]
    #             else:
    #                 yield self.__class__(mol, graph=self.G)

    #     if self.GetNumAtoms() >= config.min_atoms:
    #         for radical in build_radicals(self):
    #             yield self.__class__(radical, graph=self.G, terminal=True)


def build_radicals(starting_mol):
    """Build organic radicals. """
    
    generated_smiles = set()
    
    for i, atom in enumerate(starting_mol.GetAtoms()):
        if get_free_valence(atom) > 0:
            rw_mol = rdkit.Chem.RWMol(starting_mol)
            rw_mol.GetAtomWithIdx(i).SetNumRadicalElectrons(1)
            
            Chem.SanitizeMol(rw_mol)            
            smiles = Chem.MolToSmiles(rw_mol)
            if smiles not in generated_smiles:
                 # This makes sure the atom ordering is standardized
                yield Chem.MolFromSmiles(smiles) 
                generated_smiles.add(smiles)
