from abc import abstractmethod
from typing import (
    Iterable,
    Iterator,
    Optional,
    )

from networkx import DiGraph
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import (
    MolFromSmiles,
    MolToSmiles,
    )

from alphazero.nodes.graph_node import GraphNode
from molecule_graph.molecule_tools import (
    build_molecules,
    build_radicals,
    )


class MoleculeNode(GraphNode):
    
    def __init__(self, parent: any, molecule: Mol) -> None:
        self._parent: any = parent
        self._molecule: Mol = molecule
        self._smiles = MolToSmiles(self._molecule)
    
    def __eq__(self, other: any) -> bool:
        return self._smiles == other._smiles
    
    def __hash__(self) -> int:
        return hash(self._smiles)
    
    def __repr__(self) -> str:
        return self._smiles
    
    def get_successors(self) -> Iterable['MoleculeNode']:
        # TODO: should these functions be brought into this class?
        num_atoms = self.molecule.GetNumAtoms()
        if num_atoms < self._parent.config.max_atoms:
            yield from (MoleculeNode(self.parent, molecule)
                        for molecule in build_molecules(self._molecule, **self._parent.config.build_kwargs))
        
        if num_atoms >= self._parent.config.min_atoms:
            yield from (MoleculeNode(self.parent, molecule)
                        for molecule in build_radicals(self._molecule))
    
    @property
    def smiles(self) -> str:
        return self._smiles
    
    @property
    def molecule(self) -> Mol:
        return self._molecule
    
    @property
    def parent(self) -> any:
        return self._parent
