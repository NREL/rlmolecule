from typing import Iterable

from rdkit.Chem.rdchem import Mol

from molecule_game.molecule_node import MoleculeNode
from molecule_game.molecule_tools import (
    build_molecules,
    build_radicals,
    )


class StableRadicalOptimizationNode(MoleculeNode):
    """
    A MoleculeNode implemenation which uses simple transformations (such as adding a bond) to define
    a graph over molecular structures.
    """
    
    def __init__(self, parent: 'StableRadicalOptimizationGame', molecule: Mol, is_radical: bool) -> None:
        self._parent: 'StableRadicalOptimizationGame' = parent
        self._is_radical: bool = is_radical
        super().__init__(molecule)
    
    @property
    def is_radical(self) -> bool:
        return self._is_radical
    
    def get_successors(self) -> Iterable['StableRadicalOptimizationNode']:
        if self.is_radical:
            return []  # radicals are leaf nodes for this problem
        
        # TODO: should these functions be brought into this class?
        num_atoms = self.molecule.GetNumAtoms()
        parent = self._parent
        molecule = self._molecule
        
        if num_atoms < self._parent.config.max_atoms:
            bml = list((StableRadicalOptimizationNode(parent, molecule, False)
                        for molecule in build_molecules(molecule, **parent.config.build_kwargs)))
            yield from bml
            # yield from (MoleculeNode(parent, molecule)
            #             for molecule in build_molecules(molecule, **parent.config.build_kwargs))
        
        if num_atoms >= self._parent.config.min_atoms:
            br = list((StableRadicalOptimizationNode(parent, molecule, True)
                       for molecule in build_radicals(molecule)))
            yield from br
            # yield from (MoleculeNode(parent, molecule)
            #             for molecule in build_radicals(molecule))
    
    @property
    def parent(self) -> any:
        return self._parent
