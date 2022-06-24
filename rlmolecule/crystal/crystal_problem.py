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
                 features: int = 64,
                 num_heads: int = 4,
                 num_messages: int = 3,
                 **kwargs) -> None:
        self.num_messages = num_messages
        self.num_heads = num_heads
        self.features = features
        self.preprocessor = preprocessor if preprocessor is not None else CrystalPreprocessor()
        # Each element has a stoichiometry, so just multiply by 2
        self.num_eles_and_stoich = 2 * self.preprocessor.max_num_elements
        # This is the size of the vocabulary for the elements,
        # as well as each element with its stoichiometry
        # (e.g., Li1Sc1F4 would be Li, Li1, Sc, Sc1, F, F4),
        # UPDATE 20220614: just use a mapping for the elements,
        # and keep the stoichiometry separate
        # e.g., Li1Sc1F4 would be Li, 1, Sc, 1, F, 4)
        self.num_eles_and_stoich_comb = (len(self.preprocessor.elements)
                                         + (len(self.preprocessor.elements)
                                            * self.preprocessor.max_stoich))
        self.num_crystal_sys = len(self.preprocessor.crystal_systems)
        self.num_proto_strc = len(self.preprocessor.proto_strc_names)
        self.num_proto_eles = self.preprocessor.max_ele_Z
        self.num_proto_ele_replacements = (len(self.preprocessor.elements)
                                           * self.num_proto_eles)
        super(CrystalTFAlphaZeroProblem, self).__init__(engine=engine, **kwargs)

    def policy_model(self,
                     features: int = 64,
                     ) -> tf.keras.Model:
        """ Constructs a policy model that predicts value, pi_logits from a batch of crystal inputs. Main model used in
        policy training and loading weights

        :param features: Size of network hidden layers
        :return: The constructed policy model
        """
        # Define inputs
        # 5 conducting ions, 8 anions, 17 framework cations, up to 5 elements in a composition.
        # I will include the elements by themselves,
        # and the elements with a stoichiometry e.g., 'Cl', 'Cl6'
        # TODO Many element stoichiometries are not present.
        # For now I will just include all of them
        # There are up to 10 items here:
        # 1 conducting ion, 2 anions, 2 framework cations, and a stoichiometry for each
        # e.g., for K1La1Sb2I2N4: K, 1, La, 1, Sb, 2, I, 2, N, 4
        element_class = layers.Input(shape=[self.num_eles_and_stoich],
                                     dtype=tf.int64, name='eles_and_stoich')
        # 7 crystal systems
        crystal_sys_class = layers.Input(shape=[1],
                                         dtype=tf.int64, name='crystal_sys')
        # 4170 total prototype structures
        proto_strc_class = layers.Input(shape=[1],
                                        dtype=tf.int64, name='proto_strc')
        proto_ele_class = layers.Input(shape=[self.preprocessor.max_num_elements],
                                       dtype=tf.int64, name='proto_ele_replacements')

        input_tensors = [element_class,
                         crystal_sys_class,
                         proto_strc_class,
                         proto_ele_class,
                         ]

        element_embedding = layers.Embedding(
            input_dim=self.num_eles_and_stoich_comb + 1, output_dim=features,
            input_length=self.num_eles_and_stoich, mask_zero=True,
            name='element_embedding')(element_class)
        # sum the embeddings of each of the elements
        element_embedding = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=-2, keepdims=True),
                                          output_shape=lambda s: (s[-1],))(element_embedding)
        element_embedding = layers.Reshape((-1, features))(element_embedding)

        crystal_sys_embedding = layers.Embedding(
            input_dim=self.num_crystal_sys + 1, output_dim=features,
            input_length=1, mask_zero=True,
            name='crystal_sys_embedding')(crystal_sys_class)
        proto_strc_embedding = layers.Embedding(
            input_dim=self.num_proto_strc + 1, output_dim=features,
            input_length=1, mask_zero=True,
            name='proto_strc_embedding')(proto_strc_class)

        # also add the element replacements
        proto_ele_embedding = layers.Embedding(
            input_dim=self.num_proto_ele_replacements + 1, output_dim=features,
            input_length=self.preprocessor.max_num_elements, mask_zero=True,
            name='proto_ele_embedding')(proto_ele_class)
        # sum the embeddings of each of the elements
        proto_ele_embedding = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=-2, keepdims=True),
                                            output_shape=lambda s: (s[-1],))(proto_ele_embedding)
        proto_ele_embedding = layers.Reshape((-1, features))(proto_ele_embedding)

        # Merge all available features into a single large vector via concatenation
        x = layers.concatenate([element_embedding,
                                crystal_sys_embedding,
                                proto_strc_embedding,
                                proto_ele_embedding])

        # pass the features through a couple of dense layers
        x = layers.Dense(features, activation='relu')(x)
        x = layers.Dense(features // 2, activation='relu')(x)

        # output a single prediction value, and a prior logit at the end
        value_logit = layers.Dense(1)(x)
        pi_logit = layers.Dense(1)(x)

        return tf.keras.Model(input_tensors, [value_logit, pi_logit], name='policy_model')

    def get_policy_inputs(self, state: CrystalState) -> Dict:
        return self.preprocessor.construct_feature_matrices(state)
