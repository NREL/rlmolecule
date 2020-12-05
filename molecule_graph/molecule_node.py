from abc import abstractmethod
from typing import (
    Iterator,
    Optional,
    )

from networkx import DiGraph
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import (
    MolFromSmiles,
    MolToSmiles,
    )

from molecule_graph.molecule_tools import (
    build_molecules,
    build_radicals,
    )
from alphazero.nodes.networkx_node import NetworkXNode


class MoleculeNode(NetworkXNode):
    
    @staticmethod
    def make_from_SMILES(parent: any, smiles: str, networkx_graph: Optional[DiGraph] = None) -> 'MoleculeNode':
        """
        Factory method for making a MoleculeNode from a SMILES string. If networkx_graph is None, a new one will be
        created
        """
        if networkx_graph is None:
            networkx_graph = DiGraph()
        return MoleculeNode(parent, networkx_graph, MolFromSmiles(smiles))
    
    def __init__(self, parent: any, networkx_graph: DiGraph, molecule: Mol) -> None:
        self._parent: any = parent
        self._molecule: Mol = molecule
        self._smiles = MolToSmiles(self._molecule)
        super().__init__(networkx_graph)
    
    def _eq(self, other: any) -> bool:
        return self.smiles == other.smiles
    
    def _repr(self) -> str:
        return self.smiles
    
    @property
    def smiles(self) -> str:
        return self._smiles
    
    @property
    def molecule(self) -> Mol:
        return self._molecule
    
    @property
    def parent(self) -> any:
        return self._parent
    
    @abstractmethod
    @property
    def reward(self) -> float:
        pass
    
    def _expand(self) -> Iterator['MoleculeNode']:
        # TODO: should these functions be brought into this class?
        num_atoms = self.molecule.GetNumAtoms()
        if num_atoms < self._parent.config.max_atoms:
            yield from MoleculeNode(
                self.parent,
                self.graph,
                build_molecules(self, **self._parent.build_kwargs))
        
        if num_atoms >= self._parent.config.min_atoms:
            yield from MoleculeNode(
                self.parent,
                self.graph,
                build_radicals(self))
    
    def __hash__(self) -> int:
        return hash(self.smiles)
