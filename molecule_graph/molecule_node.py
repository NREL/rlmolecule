from abc import abstractmethod
from typing import Iterator

from networkx import DiGraph
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles

from molecule_graph.molecule_tools import (
    build_molecules,
    build_radicals,
    )
from alphazero.nodes.networkx_node import NetworkXNode


class MoleculeNode(NetworkXNode):
    
    def __init__(self, config: any, networkx_graph: DiGraph, molecule: Mol) -> None:
        super().__init__(networkx_graph)
        self._config: any = config
        self._molecule: Mol = molecule
        self._smiles = MolToSmiles(self._molecule)
    
    @property
    def smiles(self):
        return self._smiles
    
    @property
    def molecule(self):
        return self._molecule
    
    def __hash__(self) -> int:
        return hash(self.smiles)
    
    def __eq__(self, other: any) -> bool:
        return self.smiles == other.smiles
    
    def __repr__(self) -> str:
        return self.smiles
    
    def _expand(self) -> Iterator['MoleculeNode']:
        # TODO: should these functions be brought into this class?
        num_atoms = self.molecule.GetNumAtoms()
        if num_atoms < self._config.max_atoms:
            yield from build_molecules(self, **self._config.build_kwargs)
        
        if num_atoms >= self._config.min_atoms:
            yield from build_radicals(self)
    
    @abstractmethod
    @property
    def reward(self) -> float:
        pass
