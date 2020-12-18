import rdkit
from rdkit.Chem.QED import qed

from molecule_game.molecule_config import MoleculeConfig
from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.molecule.molecule_state import MoleculeState


class QEDOptimizationProblem(MCTSProblem):

    def __init__(self, config: MoleculeConfig) -> None:
        self.__config = config

    def get_initial_state(self) -> MoleculeState:
        return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self.__config)

    def compute_reward(self, state: MoleculeState) -> float:
        if state.is_terminal:
            return qed(state.molecule)
        return 0.0
