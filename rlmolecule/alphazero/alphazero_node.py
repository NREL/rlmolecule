import io
import itertools
import logging

from typing import (
    List,
    Optional,
)

import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences

from rlmolecule.alphazero.alphazero_game import AlphaZeroGame
from rlmolecule.mcts.mcts_node import MCTSNode
from rlmolecule.tree_search.tree_search_state import TreeSearchState

logger = logging.getLogger(__name__)


class AlphaZeroNode(MCTSNode):
    """
    A class which implements the AlphaZero search methodology, with the assistance of a supplied
    AlphaZeroGame implementation ("game").
    """

    def __init__(self, state: TreeSearchState, game: AlphaZeroGame) -> None:
        super().__init__(state, game)

        self._prior_logit: float = np.nan
        self._child_priors: Optional[List['AlphaZeroNode']] = None  # lazily initialized
        self._policy_inputs: Optional[{}] = None  # lazily initialized
        self._policy_data = None  # lazily initialized

    def evaluate(self):
        """
        For a given node, build the children, add them to the graph, and run the
        policy network to get prior_logits and a value.

        Returns:
        value (float): the estimated value of `parent`.
        """
        # Looks like in rlmolecule, we always expand, even if this is the
        # first time we've visited the node
        if self.terminal:
            return self.reward

        children = self.successors

        # Handle the case where a node doesn't have any children
        if self.terminal:
            return self.reward

        # Run the policy network to get value and prior_logit predictions
        # noinspection PyUnresolvedReferences
        values, prior_logits = self._game.problem.policy_predictions(self.policy_inputs_with_children())

        # inputs to policy network
        prior_logits = prior_logits[1:].numpy().flatten()

        # Update child nodes with predicted prior_logits
        for child, prior_logit in zip(children, prior_logits):
            child._prior_logit = float(prior_logit)

        # Return the parent's predicted value
        return float(tf.nn.sigmoid(values[0]))

    def child_prior(self, child: 'AlphaZeroNode') -> float:
        """
        Prior probabilities (unlike logits) depend on the parent
        """
        return self.child_priors[child]

    @property
    def child_priors(self) -> {'AlphaZeroNode': float}:
        """
        Get a list of priors for the node's children, with optionally added dirichlet noise.
        Caches the result, so the noise is consistent.
        """
        if self._child_priors is None:
            # Perform the softmax over the children node's prior logits
            children: [AlphaZeroNode] = self.successors
            priors = tf.nn.softmax([child._prior_logit for child in children]).numpy()

            # Add the optional exploration noise
            # noinspection PyTypeChecker
            game: AlphaZeroGame = self._game
            if game.dirichlet_noise:
                random_state = np.random.RandomState()
                noise = random_state.dirichlet(
                    np.ones_like(priors) * game.dirichlet_alpha)

                priors = priors * (1 - game.dirichlet_x) + (noise * game.dirichlet_x)

            assert np.isclose(priors.sum(), 1.)  # Just a sanity check
            self._child_priors = {child: prior for child, prior in zip(children, priors)}

        return self._child_priors

    @property
    def policy_inputs(self):
        """
        :return GNN inputs for the node
        """
        if self._policy_inputs is None:
            # noinspection PyUnresolvedReferences
            self._policy_inputs = self._game.problem.construct_feature_matrices(self)
        return self._policy_inputs

    def policy_inputs_with_children(self) -> {}:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """

        policy_inputs = [node.policy_inputs for node in itertools.chain((self,), self.successors)]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}

    def store_policy_data(self):
        data = self.policy_inputs_with_children()
        visit_counts = np.array([child.visits for child in self.successors])
        data['visit_probs'] = visit_counts / visit_counts.sum()

        with io.BytesIO() as f:
            np.savez_compressed(f, **data)
            self._policy_data = f.getvalue()

    @property
    def policy_data(self):
        if self._policy_data is None:
            self.store_policy_data()
        return self._policy_data

    def _make_successor(self, action: TreeSearchState) -> 'AlphaZeroNode':
        # noinspection PyTypeChecker
        return AlphaZeroNode(action, self._game)
