import itertools
import logging
from functools import partial
from typing import Dict, Optional

import nfp
import sqlalchemy
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.molecule.molecule_state import MoleculeState
from rlmolecule.molecule.policy.model import build_policy_trainer
from rlmolecule.molecule.policy.preprocessor import MolPreprocessor, load_preprocessor

logger = logging.getLogger(__name__)


class MoleculeAlphaZeroProblem(AlphaZeroProblem):

    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 preprocessor: Optional[MolPreprocessor] = None,
                 preprocessor_data: Optional[str] = None,
                 policy_checkpoint_dir: Optional[str] = None,
                 features: int = 64,
                 num_heads: int = 4,
                 num_messages: int = 3,
                 ) -> None:

        super(MoleculeAlphaZeroProblem, self).__init__(engine)
        self.preprocessor = preprocessor if preprocessor else load_preprocessor(preprocessor_data)
        policy_model = build_policy_trainer(self.preprocessor, features, num_heads, num_messages)

        if policy_checkpoint_dir:
            latest = tf.train.latest_checkpoint(policy_checkpoint_dir)
            if latest:
                policy_model.load_weights(latest)
                logger.info(f'Loaded checkpoint {latest}')
            else:
                logger.info('No checkpoint found')

        policy_model_layer = policy_model.layers[-1].policy_model
        self.policy_evaluator = tf.function(experimental_relax_shapes=True)(policy_model_layer.predict_step)

    def get_network_inputs(self, state: MoleculeState) -> Dict:
        return self.preprocessor.construct_feature_matrices(state.molecule)

    def get_batched_network_inputs(self, parent: AlphaZeroVertex) -> Dict:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """

        policy_inputs = [self.get_network_inputs(vertex.state)
                         for vertex in itertools.chain((parent,), parent.children)]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}

    def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):

        values, prior_logits = self.policy_evaluator(self.get_batched_network_inputs(parent))

        # inputs to policy network
        priors = tf.nn.softmax(prior_logits[1:]).numpy().flatten()

        # Update child nodes with predicted prior_logits
        children_priors = {vertex: prior for vertex, prior in zip(parent.children, priors)}
        value = float(tf.nn.sigmoid(values[0]))

        return value, children_priors

    def get_network_inputs_from_serialized_parent(
            self,
            serialized_parent: tf.Tensor) -> ({}, {}, {}):

        parent = MoleculeState.deserialize(serialized_parent.numpy().decode())

        policy_inputs = [self.get_network_inputs(parent)
                         for state in itertools.chain((parent,), parent.get_next_actions())]

        policy_inputs = {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                         for key in policy_inputs[0].keys()}

        return policy_inputs['atom'], policy_inputs['bond'], policy_inputs['connectivity']

    def create_dataset(self) -> tf.data.Dataset:
        """
        Creates a tensorflow dataset pipeline to batch game positions from the replay buffer into

        :param problem:
        :return:
        """

        def get_policy_inputs_tf(parent, visit_probabilities, reward,
                                 problem: MoleculeAlphaZeroProblem) -> {}:
            atom, bond, connectivity = tf.py_function(
                problem.get_network_inputs_from_serialized_parent, inp=[parent],
                Tout=[tf.int64, tf.int64, tf.int64])

            atom.set_shape([None, None])
            bond.set_shape([None, None])
            connectivity.set_shape([None, None, 2])
            return {'atom': atom, 'bond': bond, 'connectivity': connectivity}, (visit_probabilities, reward)

        dataset = tf.data.Dataset.from_generator(
            self.iter_recent_games,
            output_shapes=((), (None,), ()),
            output_types=(tf.string, tf.float32, tf.float32)) \
            .repeat() \
            .shuffle(self.max_buffer_size) \
            .map(partial(get_policy_inputs_tf, problem=self),
                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .padded_batch(self.batch_size,
                          padding_values=({'atom': nfp.zero, 'bond': nfp.zero, 'connectivity': nfp.zero},
                                          (0., 0.))) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
