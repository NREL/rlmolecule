from typing import Iterable

from rdkit.Chem.rdchem import Mol

from molecule_game.molecule_node import MoleculeNode
from molecule_game.molecule_tools import (
    build_molecules,
    build_radicals,
    )


class StableRadicalOptimizationNode(MoleculeNode):
    
    def __init__(self, parent: 'MoleculeGame', molecule: Mol, is_radical: bool) -> None:
        self._is_radical: bool = is_radical
        super().__init__(parent, molecule)
    
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
            print("{} bml: {}".format(self, bml))
            yield from bml
            # yield from (MoleculeNode(parent, molecule)
            #             for molecule in build_molecules(molecule, **parent.config.build_kwargs))
        
        if num_atoms >= self._parent.config.min_atoms:
            br = list((StableRadicalOptimizationNode(parent, molecule, True)
                       for molecule in build_radicals(molecule)))
            print("{} br: {}".format(self, br))
            yield from br
            # yield from (MoleculeNode(parent, molecule)
            #             for molecule in build_radicals(molecule))
