import itertools
import logging
from typing import Dict, List, Optional

import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from rlmolecule.alphazero.alphazero_game import AlphaZeroGame
from rlmolecule.alphazero.alphazero_node import AlphaZeroNode
from rlmolecule.molecule.policy.model import build_policy_evaluator
from rlmolecule.molecule.policy.preprocessor import load_preprocessor
from rlmolecule.state import State

logger = logging.getLogger(__name__)


class MoleculeAlphaZeroGame(AlphaZeroGame):
    def __init__(self,
                 *args,
                 preprocessor=None,
                 preprocessor_data=None,
                 policy_checkpoint_dir=None,
                 **kwargs):
        super(MoleculeAlphaZeroGame, self).__init__(*args, **kwargs)

        self.preprocessor = preprocessor if preprocessor else load_preprocessor(preprocessor_data)
        self.policy_evaluator, loaded_checkpoint = build_policy_evaluator(policy_checkpoint_dir)

        if loaded_checkpoint:
            logger.info(f'{self.id}: Loaded checkpoint {loaded_checkpoint}')
        else:
            logger.info(f'{self.id}: No checkpoint found {loaded_checkpoint}')


class MoleculeAlphaZeroNode(AlphaZeroNode):
    def __init__(self, state: State, game: Optional[MoleculeAlphaZeroGame] = None) -> None:

        if game is None:
            game = MoleculeAlphaZeroGame()

        super(MoleculeAlphaZeroNode, self).__init__(state, game)

    def get_policy_inputs(self) -> Dict:
        """
        :return GNN inputs for the node
        """
        if self._policy_inputs is None:
            self._policy_inputs = self.game.preprocessor.construct_feature_matrices(self)
        return self._policy_inputs

    def get_batched_policy_inputs(self, successors) -> Dict:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """

        policy_inputs = [node.get_policy_inputs() for node in itertools.chain((self,), successors)]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}

    def policy(self, successors: List['MCTSNode']) -> (float, Dict['MCTSNode', float]):

        values, prior_logits = self.game.policy_evaluator(self.get_batched_policy_inputs(successors))

        # inputs to policy network
        priors = tf.nn.softmax(prior_logits[1:]).numpy().flatten()

        # Update child nodes with predicted prior_logits
        successor_priors = {node: prior for node, prior in zip(successors, priors)}
        value = float(tf.nn.sigmoid(values[0]))

        return value, successor_priors

    # def store_policy_data(self):
    #     data = self.policy_inputs_with_children()
    #     visit_counts = np.array([child.visits for child in self.successors])
    #     data['visit_probs'] = visit_counts / visit_counts.sum()
    #
    #     with io.BytesIO() as f:
    #         np.savez_compressed(f, **data)
    #         self._policy_data = f.getvalue()
    #
    # @property
    # def policy_data(self):
    #     if self._policy_data is None:
    #         self.store_policy_data()
    #     return self._policy_data
