from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import (
    MolToSmiles,
    )

from alphazero.graph_node import GraphNode


class MoleculeNode(GraphNode):
    """
    An abstract GraphNode implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES string is also stored for
    efficient hashing.
    """
    
    def __init__(self, molecule: Mol) -> None:
        self._molecule: Mol = molecule
        self._smiles = MolToSmiles(self._molecule)
    
    def __repr__(self) -> str:
        """
        delegates to the SMILES string
        """
        return self._smiles
    
    def equals(self, other: 'MoleculeNode') -> bool:
        """
        delegates to the SMILES string
        """
        return isinstance(other, 'MoleculeNode') and self._smiles == other._smiles
    
    def hash(self) -> int:
        """
        delegates to the SMILES string
        """
        return hash(self._smiles)
    
    @property
    def smiles(self) -> str:
        return self._smiles
    
    @property
    def molecule(self) -> Mol:
        return self._molecule
