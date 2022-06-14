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
                 *args, **kwargs):
        super(CrystalProblem, self).__init__(*args, **kwargs)

    def get_initial_state(self) -> CrystalState:
        # The root node in the action space is the string 'root'
        action_node = "root"
        return CrystalState(action_node)


class CrystalTFAlphaZeroProblem(CrystalProblem, TFAlphaZeroProblem, ABC):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 preprocessor: Optional[CrystalPreprocessor] = None,
                 preprocessor_data: Optional[str] = None,
                 features: int = 64,
                 num_heads: int = 4,
                 num_messages: int = 3,
                 actions_to_ignore: Optional[set] = None,
                 **kwargs) -> None:
        self.num_messages = num_messages
        self.num_heads = num_heads
        self.features = features
        self.preprocessor = preprocessor if preprocessor is not None else CrystalPreprocessor()
        self.actions_to_ignore = actions_to_ignore
        super(CrystalTFAlphaZeroProblem, self).__init__(engine=engine, **kwargs)

    def policy_model(self,
                     features: int = 64,
                     num_eles_and_stoich: int = 10,
                     num_eles_and_stoich_comb: int = 252,
                     num_crystal_sys: int = 7,
                     num_proto_strc: int = 4170,
                     ) -> tf.keras.Model:
        """ Constructs a policy model that predicts value, pi_logits from a batch of crystal inputs. Main model used in
        policy training and loading weights

        :param features: Size of network hidden layers
        :return: The constructed policy model
        """
        # Define inputs
        # 5 conducting ions, 8 anions, 17 framework cations, up to 8 elements in a composition.
        # I will include the elements by themselves, and the elements with a stoichiometry e.g., 'Cl', 'Cl6'
        # TODO Many element stoichiometries are not present. For now I will just include all of them
        # There are up to 10 items here:
        # 1 conducting ion, 2 anions, 2 framework cations, and a stoichiometry for each
        # e.g., for K1La1Sb2I2N4: K, K1, La, La1, Sb, Sb2, I, I2, N, N4
        element_class = layers.Input(shape=[num_eles_and_stoich], dtype=tf.int64, name='eles_and_stoich')
        # 7 crystal systems
        crystal_sys_class = layers.Input(shape=[1], dtype=tf.int64, name='crystal_sys')
        # 4170 total prototype structures
        proto_strc_class = layers.Input(shape=[1], dtype=tf.int64, name='proto_strc')

        input_tensors = [element_class, crystal_sys_class, proto_strc_class]

        element_embedding = layers.Embedding(
            input_dim=num_eles_and_stoich_comb + 1, output_dim=features,
            input_length=num_eles_and_stoich, mask_zero=True, name='element_embedding')(element_class)
        # sum the embeddings of each of the elements
        element_embedding = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=-2, keepdims=True),
                                          output_shape=lambda s: (s[-1],))(element_embedding)
        element_embedding = layers.Reshape((-1, features))(element_embedding)

        crystal_sys_embedding = layers.Embedding(
            input_dim=num_crystal_sys + 1, output_dim=features,
            input_length=1, mask_zero=True, name='crystal_sys_embedding')(crystal_sys_class)
        proto_strc_embedding = layers.Embedding(
            input_dim=num_proto_strc + 1, output_dim=features,
            input_length=1, mask_zero=True, name='proto_strc_embedding')(proto_strc_class)

        # Merge all available features into a single large vector via concatenation
        x = layers.concatenate([element_embedding, crystal_sys_embedding, proto_strc_embedding])

        # pass the features through a couple of dense layers
        x = layers.Dense(features, activation='relu')(x)
        x = layers.Dense(features // 2, activation='relu')(x)

        # output a single prediction value, and a prior logit at the end
        value_logit = layers.Dense(1)(x)
        pi_logit = layers.Dense(1)(x)

        return tf.keras.Model(input_tensors, [value_logit, pi_logit], name='policy_model')

    def get_policy_inputs(self, state: CrystalState) -> Dict:
        return self.preprocessor.construct_feature_matrices(state)
