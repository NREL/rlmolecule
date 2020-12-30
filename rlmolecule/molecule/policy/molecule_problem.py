import itertools
import logging
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.molecule.policy.model import build_policy_evaluator
from rlmolecule.molecule.policy.preprocessor import MolPreprocessor, load_preprocessor

logger = logging.getLogger(__name__)


class MoleculeAlphaZeroProblem(AlphaZeroProblem):

    def __init__(self,
                 preprocessor: Optional[MolPreprocessor] = None,
                 preprocessor_data: Optional[str] = None,
                 policy_checkpoint_dir: Optional[str] = None
                 ) -> None:

        super(MoleculeAlphaZeroProblem, self).__init__()
        self.preprocessor = preprocessor if preprocessor else load_preprocessor(preprocessor_data)
        self.policy_evaluator, loaded_checkpoint = build_policy_evaluator(policy_checkpoint_dir)

        if loaded_checkpoint:
            logger.info(f'{self.id}: Loaded checkpoint {loaded_checkpoint}')
        else:
            logger.info(f'{self.id}: No checkpoint found {loaded_checkpoint}')

    def get_network_inputs(self, vertex: AlphaZeroVertex) -> Dict:
        return self.preprocessor.construct_feature_matrices(vertex.state.molecule)

    def get_batched_network_inputs(self, parent: AlphaZeroVertex) -> Dict:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """

        policy_inputs = [self.get_network_inputs(vertex) for vertex in itertools.chain((parent,), parent.children)]
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
