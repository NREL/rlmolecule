import numpy as np
import rdkit
from rdkit.Chem.QED import qed

from molecule_game.molecule_config import MoleculeConfig
from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.molecule.molecule_state import MoleculeState


class QEDOptimizationProblem(AlphaZeroProblem):

    def __init__(self, config: MoleculeConfig) -> None:
        self.__config = config

    def get_initial_state(self) -> MoleculeState:
        return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self.__config)

    def get_reward(self, state: MoleculeState) -> float:
        if state.forced_terminal:
            return qed(state.molecule)
        return 0.0

    def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
        random_state = np.random.RandomState()
        children = parent.children
        priors = random_state.dirichlet(np.ones(len(children)))

        return random_state.random(), {vertex: prior for vertex, prior in zip(children, priors)}
