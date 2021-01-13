import rdkit
from rdkit.Chem.QED import qed
import numpy as np
from sqlalchemy import create_engine

from rlmolecule.molecule.molecule_config import MoleculeConfig
from rlmolecule.molecule.molecule_problem import MoleculeAlphaZeroProblem
from rlmolecule.molecule.molecule_state import MoleculeState



# todo: ranked rewards

class QEDOptimizationProblem(MoleculeAlphaZeroProblem):

    def __init__(self,
                 engine: 'sqlalchemy.engine.Engine',
                 config: 'MoleculeConfig', **kwargs) -> None:
        super(QEDOptimizationProblem, self).__init__(engine, **kwargs)
        self._config = config

    def get_initial_state(self) -> MoleculeState:
        return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self._config)

    def get_reward(self, state: MoleculeState) -> (float, {}):
        if state.forced_terminal:
            return qed(state.molecule), {'forced_terminal': True}
        return 0.0, {'forced_terminal': False}

config = MoleculeConfig(max_atoms=4,
                        min_atoms=1,
                        tryEmbedding=False,
                        sa_score_threshold=None,
                        stereoisomers=False)

engine = create_engine(f'sqlite:///qed_data.db',
                       connect_args={'check_same_thread': False})

problem = QEDOptimizationProblem(
    engine,
    config,
    features=8,
    num_heads=2,
    num_messages=1,
    policy_checkpoint_dir='policy_checkpoints')


