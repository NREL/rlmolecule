from rdkit.Chem.QED import qed
from rlmolecule.molecule_state import MoleculeState


class QEDState(MoleculeState):
    @property
    def reward(self) -> float:
        if self.forced_terminal:
            reward = qed(self.molecule)
            self.data.log_reward([self.smiles, reward])
            return reward

        else:
            return 0.0
