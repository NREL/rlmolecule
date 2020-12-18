from typing import Iterable

from rdkit.Chem.rdchem import Mol

from molecule_game.molecule_config import MoleculeConfig
from rlmolecule.molecule.molecule_state import MoleculeState
from rlmolecule.molecule.molecule_tools import (
    build_molecules,
    build_radicals,
)


class StableRadicalOptimizationState(MoleculeState):
    """
    A MoleculeNode implemenation which uses simple transformations (such as adding a bond) to define
    a graph over molecular structures.
    """

    def __init__(self, molecule: Mol, config: MoleculeConfig, is_radical: bool) -> None:
        super().__init__(molecule, config)
        self._is_radical: bool = is_radical

    @property
    def is_radical(self) -> bool:
        return self._is_radical

    def get_next_actions(self) -> Iterable['StableRadicalOptimizationState']:
        if self.is_radical:
            return []  # radicals are leaf nodes for this problem

        # TODO: should these functions be brought into this class?
        num_atoms = self.molecule.GetNumAtoms()
        config = self.config

        if num_atoms < config.max_atoms:
            yield from (StableRadicalOptimizationState(molecule, config, False)
                        for molecule in build_molecules(self.molecule, **config.build_kwargs))

        if num_atoms >= config.min_atoms:
            yield from (StableRadicalOptimizationState(molecule, config, True)
                        for molecule in build_radicals(self.molecule))
