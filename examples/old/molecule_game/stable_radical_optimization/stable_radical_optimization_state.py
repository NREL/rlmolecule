from typing import Iterable

from rdkit.Chem.rdchem import Mol

from rlmolecule.molecule.molecule_state import MoleculeState
from rlmolecule.molecule.molecule_building import (
    build_molecules,
    build_radicals,
)


class StableRadicalOptimizationState(MoleculeState):
    """
    A MoleculeNode implementation which uses simple transformations (such as adding a bond) to define
    a graph over molecular structures.
    """

    def __init__(self, molecule: Mol, config: any, force_terminal: bool) -> None:
        super().__init__(molecule, config, force_terminal)

    @property
    def is_radical(self) -> bool:
        return self._forced_terminal

    def get_next_actions(self) -> Iterable['StableRadicalOptimizationState']:
        # TODO: should these functions be brought into this class?
        config = self.config
        result = []
        if not self._forced_terminal:
            if self.num_atoms < config.max_atoms:
                result.extend((StableRadicalOptimizationState(molecule, config, False)
                               for molecule in build_molecules(self.molecule, **config.build_kwargs)))

                if self.num_atoms >= config.min_atoms:
                    result.extend((StableRadicalOptimizationState(molecule, config, True)
                                   for molecule in build_radicals(self.molecule)))

        return result
