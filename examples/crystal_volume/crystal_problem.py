import logging
from abc import ABC
from typing import Dict, Optional

import rdkit
import sqlalchemy

#from rlmolecule.alphazero.tensorflow.tfalphazero_problem import TFAlphaZeroProblem
from rlmolecule.mcts.mcts_problem import MCTSProblem
#from rlmolecule.molecule.molecule_state import MoleculeState
from examples.crystal_volume.crystal_state import CrystalState
#from rlmolecule.molecule.policy.model import policy_model

logger = logging.getLogger(__name__)


class CrystalProblem(MCTSProblem, ABC):
    def __init__(self, builder: 'CrystalBuilder', *args, **kwargs):
        self._config = builder
        super(CrystalProblem, self).__init__(*args, **kwargs)

    def get_initial_state(self) -> CrystalState:
        elements = ()
        # The root node in the action space is an empty string
        action_node = ""
        return CrystalState(elements, action_node, self._config)


# TODO implement for crystals
# class MoleculeTFAlphaZeroProblem(MoleculeProblem, TFAlphaZeroProblem, ABC):
#     def __init__(self,
#                  engine: sqlalchemy.engine.Engine,
#                  builder: 'MoleculeBuilder',
#                  preprocessor: Optional[MolPreprocessor] = None,
#                  preprocessor_data: Optional[str] = None,
#                  features: int = 64,
#                  num_heads: int = 4,
#                  num_messages: int = 3,
#                  **kwargs) -> None:
#         self.num_messages = num_messages
#         self.num_heads = num_heads
#         self.features = features
#         self.preprocessor = preprocessor if preprocessor else load_preprocessor(preprocessor_data)
#         super(MoleculeTFAlphaZeroProblem, self).__init__(builder=builder, engine=engine, **kwargs)
#
#     def policy_model(self) -> 'tf.keras.Model':
#         return policy_model(self.preprocessor,
#                             features=self.features,
#                             num_heads=self.num_heads,
#                             num_messages=self.num_messages)
#
#     def get_policy_inputs(self, state: MoleculeState) -> Dict:
#         return self.preprocessor.construct_feature_matrices(state.molecule)
