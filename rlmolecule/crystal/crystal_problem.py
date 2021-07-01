import logging
from abc import ABC
from typing import Dict, Optional

import sqlalchemy as sqlalchemy
import tensorflow as tf
from tensorflow.keras import layers

from rlmolecule.alphazero.tensorflow.tfalphazero_problem import TFAlphaZeroProblem
from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.crystal.preprocessor import CrystalPreprocessor
from rlmolecule.mcts.mcts_problem import MCTSProblem

logger = logging.getLogger(__name__)


class CrystalProblem(MCTSProblem, ABC):
    def __init__(self,
                 builder: 'CrystalBuilder',
                 *args, **kwargs):
        self._config = builder
        super(CrystalProblem, self).__init__(*args, **kwargs)

    def get_initial_state(self) -> CrystalState:
        # The root node in the action space is the string 'root'
        action_node = "root"
        return CrystalState(action_node, self._config)

class CrystalTFAlphaZeroProblem(CrystalProblem, TFAlphaZeroProblem, ABC):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 builder: 'CrystalBuilder',
                 preprocessor: Optional[CrystalPreprocessor] = None,
                 preprocessor_data: Optional[str] = None,
                 features: int = 64,
                 num_heads: int = 4,
                 num_messages: int = 3,
                 **kwargs) -> None:
        self.num_messages = num_messages
        self.num_heads = num_heads
        self.features = features
        self.preprocessor = preprocessor if preprocessor else load_preprocessor(preprocessor_data)
        super(CrystalTFAlphaZeroProblem, self).__init__(builder=builder, engine=engine, **kwargs)
#
    def policy_model(self,
                     features: int = 64,
                     num_eles_and_stoich: int = 252,
                     num_crystal_sys: int = 7,
                     num_proto_strc: int = 4170,
                     ) -> tf.keras.Model:
        """ Constructs a policy model that predicts value, pi_logits from a batch of molecule inputs. Main model used in
        policy training and loading weights

        :param preprocessor: a MolPreprocessor class for initializing the embedding matrices
        :param features: Size of network hidden layers
        :return: The constructed policy model
        """
        # Define inputs
        # 5 conducting ions, 8 anions, 17 framework cations, up to 8 elements in a composition.
        # I will include the elements by themselves, and the elements with a stoichiometry e.g., 'Cl', 'Cl6'
        # TODO Many element stoichiometries are not present. For now I will just include all of them
        element_class = layers.Input(shape=[None], dtype=tf.int64, name='eles_and_stoich')
        # 7 crystal systems
        crystal_sys_class = layers.Input(shape=[None], dtype=tf.int64, name='crystal_sys')
        # 4170 total prototype structures
        proto_strc_class = layers.Input(shape=[None], dtype=tf.int64, name='proto_strc')

        input_tensors = [element_class, crystal_sys_class, proto_strc_class]

        element_embedding = layers.Embedding(
            num_eles_and_stoich, features, name='conducting_embedding')(element_class)

        elements_output = layers.Dense(features, activation='relu')(element_embedding)

        # TODO don't need an embedding because the number of crystal systems is small(?). Just use a one-hot encoding
        crystal_sys_embedding = layers.Embedding(
            num_crystal_sys, features, name='crystal_sys_embedding')(crystal_sys_class)
        crystal_sys_output = layers.Dense(features, activation='relu')(crystal_sys_embedding)

        proto_strc_embedding = layers.Embedding(
            num_proto_strc, features, name='proto_strc_embedding')(proto_strc_class)
        proto_strc_output = layers.Dense(features, activation='relu')(proto_strc_embedding)

        # Merge all available features into a single large vector via concatenation
        x = layers.concatenate([elements_output, crystal_sys_output, proto_strc_output])
        global_state = layers.Dense(features, activation='relu')(x)
        output = layers.Dense(1)(global_state)

        return tf.keras.Model(input_tensors, output, name='policy_model')

    def get_policy_inputs(self, state: CrystalState) -> Dict:

        return self.preprocessor.construct_feature_matrices(state)
