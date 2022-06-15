from rdkit.Chem.QED import qed
from rlmolecule.molecule_state import MoleculeState


class QEDState(MoleculeState):
    @property
    def reward(self) -> float:
        if self.forced_terminal:
            return qed(self.molecule)
        else:
            return 0.0
