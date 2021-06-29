import logging
from abc import ABC

from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.mcts.mcts_problem import MCTSProblem

logger = logging.getLogger(__name__)


class CrystalProblem(MCTSProblem, ABC):
    def __init__(self, builder: 'CrystalBuilder', *args, **kwargs):
        self._config = builder
        super(CrystalProblem, self).__init__(*args, **kwargs)

    def get_initial_state(self) -> CrystalState:
        # The root node in the action space is the string 'root'
        action_node = "root"
        return CrystalState(action_node, self._config)

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
    def policy_model(self) -> 'tf.keras.Model':

        # conducting_embedding = layers.Embedding(max_framework_atoms, features, name='atom_embedding')
        # anion_embedding = layers.Embedding(max_framework_atoms, features, name='atom_embedding')
        # framework_embedding = layers.Embedding(max_framework_atoms, features, name='atom_embedding')
        #
        #
        # atom1 = conducting_embedding(input1)
        # atom2 = anion_embedding(input2)
        # atom3 = anion_embedding(input3)
        # atom4 = framework_embedding(input4)
        # atom5 = framework_embedding(input5)
        #
        # global_state = layers.Add()([atom1, atom2, atom3, atom4, atom5])
        # output = layers.Dense(some_number, activation='relu')(global_state)
        # output = layers.Dense(1)(output)

        #
#     def get_policy_inputs(self, state: MoleculeState) -> Dict:
#         return self.preprocessor.construct_feature_matrices(state.molecule)
