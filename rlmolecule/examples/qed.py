from rdkit.Chem.QED import qed
from rlmolecule.actors import CSVActorWriter
from rlmolecule.molecule_state import MoleculeState


def get_csv_logger(filename):
    return CSVActorWriter.options(
        name="qed_results", lifetime="detached", get_if_exists=True
    ).remote(filename)


class QEDState(MoleculeState):
    @property
    def reward(self) -> float:
        if self.forced_terminal:
            reward = qed(self.molecule)
            self.data.log_reward([self.smiles, reward])
            return reward

        else:
            return 0.0
