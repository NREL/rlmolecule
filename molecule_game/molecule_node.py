from abc import abstractmethod
from typing import (
    Iterable,
    )

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import (
    MolToSmiles,
    )

from alphazero.alphazero_node import AlphaZeroNode
from molecule_game.molecule_tools import (
    build_molecules,
    )


class MoleculeNode(AlphaZeroNode):
    """
    An abstract GraphNode implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES string is also stored for
    efficient hashing.
    """
    
    def __init__(self, molecule: Mol) -> None:
        super().__init__()
        self._molecule: Mol = molecule
        self._smiles = MolToSmiles(self._molecule)
    
    def __eq__(self, other: any) -> bool:
        """
        delegates to the SMILES string
        """
        return self._smiles == other._smiles
    
    def __hash__(self) -> int:
        """
        delegates to the SMILES string
        """
        return hash(self._smiles)
    
    def __repr__(self) -> str:
        """
        delegates to the SMILES string
        """
        return self._smiles
    
    @property
    def smiles(self) -> str:
        return self._smiles
    
    @property
    def molecule(self) -> Mol:
        return self._molecule
