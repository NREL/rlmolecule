import io
import itertools
import logging
from typing import (
    Iterable,
    Iterator,
    List,
    Optional,
    )

import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences

from alphazero.nodes.graph_node import GraphNode

logger = logging.getLogger(__name__)


class AlphaZeroNode(GraphNode):
    
    def __init__(self, graph_node: GraphNode, game: 'AlphaZeroGame') -> None:
        self._graph_node: GraphNode = graph_node
        self._game: 'AlphaZeroGame' = game
        self._visits: int = 0
        self._total_value: float = 0.0
        self._prior_logit: float = np.nan
        self._reward: Optional[float] = None
        self._child_priors: Optional[List['AlphaZeroNode']] = None
        self._policy_inputs: Optional[{}] = None
        self._policy_data = None
    
    def get_successors(self) -> Iterable['AlphaZeroNode']:
        game = self._game
        return (AlphaZeroNode(graph_successor, game) for graph_successor in self._graph_node.get_successors())
    
    @property
    def graph_node(self) -> GraphNode:
        return self._graph_node
    
    def compute_reward(self) -> float:
        # if self.terminal:
        #     return config.min_reward
        # TODO: more here
        pass
    
    def compute_value_estimate(self) -> float:
        """
        Used to be called "expand()".
        
        For a given node, build the chidren, add them to the graph, and run the
        policy network to get prior_logits and a value.

        Returns:
        value (float): the estimated value of `parent`.
        """
        
        children = self.get_successors_list()
        
        # Handle the case where a node doesn't have any valid children
        if len(children) == 0:
            return self._game.min_reward  # TODO: what? This doesn't make sense to me.
        
        # Run the policy network to get value and prior_logit predictions
        values, prior_logits = self._game.policy_predictions(self.policy_inputs_with_children())
        
        # inputs to policy network
        prior_logits = prior_logits[1:].numpy().flatten()
        
        # Update child nodes with predicted prior_logits
        for child, prior_logit in zip(children, prior_logits):
            child._prior_logit = float(prior_logit)
        
        # Return the parent's predicted value
        return float(tf.nn.sigmoid(values[0]))
    
    def update(self, reward: float) -> None:
        self._visits += 1
        self._total_value += reward
    
    @property
    def visits(self) -> int:
        return self._visits
    
    @property
    def value(self) -> float:
        return self._total_value / self._visits if self._visits != 0 else 0
    
    def reset_priors(self) -> None:
        self._prior_logit = np.nan
    
    def reset_updates(self) -> None:
        self._visits = 0
        self._total_value = 0
        self._reward = None
    
    def tree_policy(self) -> Iterator['AlphaZeroNode']:
        """
        Implements the tree search part of an MCTS search. Recursive function which
        returns a generator over the optimal path.
        """
        yield self
        sorted_successors = sorted(self.get_successors(), key=lambda successor: -self.ucb_score(successor))
        yield from (successor.tree_policy() for successor in sorted_successors)
    
    def mcts_step(self) -> 'AlphaZeroNode':
        """
        Perform a single MCTS step from the given starting node, including a
        tree search, expansion, and backpropagation.
        """
        
        # Perform the tree policy search
        history = list(self.tree_policy())
        leaf = history[-1]
        
        # Looks like in alphazero, we always expand, even if this is the
        # first time we've visited the node
        if not leaf.terminal:
            value = leaf.compute_value_estimate()
        else:
            value = leaf.reward
        
        # perform backprop
        for node in history:
            node.update(value)
        
        return leaf
    
    def ucb_score(self, child: 'AlphaZeroNode') -> float:
        game = self._game
        pb_c = np.log((self.visits + game.pb_c_base + 1) / game.pb_c_base) + game.pb_c_init
        pb_c *= np.sqrt(self.visits) / (child.visits + 1)
        prior_score = pb_c * self.child_prior(child)
        return prior_score + child.value
    
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
            children = self.get_successors_list()
            priors = tf.nn.softmax([child._prior_logit for child in children]).numpy()
            
            # Add the optional exploration noise
            game = self._game
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
            self._policy_inputs = self._game.construct_feature_matrices(self)
        return self._policy_inputs
    
    def policy_inputs_with_children(self) -> {}:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """
        
        policy_inputs = [node.policy_inputs for node in itertools.chain((self,), self.get_successors())]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}
    
    def store_policy_data(self):
        data = self.policy_inputs_with_children()
        visit_counts = np.array([child.visits for child in self.get_successors()])
        data['visit_probs'] = visit_counts / visit_counts.sum()
        
        with io.BytesIO() as f:
            np.savez_compressed(f, **data)
            self._policy_data = f.getvalue()
    
    @property
    def policy_data(self):
        if self._policy_data is None:
            self.store_policy_data()
        return self._policy_data
    
    @property
    def reward(self) -> float:
        assert self.terminal, "Accessing reward of non-terminal state"
        if self._reward is None:
            self._reward = self.compute_reward()
        return self._reward
    
    def softmax_sample(self) -> 'AlphaZeroNode':
        """
        Sample from successors according to their visit counts.

        Returns:
            choice: Node, the chosen successor node.
        """
        successors = self.get_successors_list()
        visit_counts = np.array([n._visits for n in successors])
        visit_softmax = tf.nn.softmax(tf.constant(visit_counts, dtype=tf.float32)).numpy()
        return successors[np.random.choice(range(len(successors)), size=1, p=visit_softmax)[0]]
    
    def run_mcts(self, num_simulations: int, explore: bool = True) -> Iterator['AlphaZeroNode']:
        """
        Performs a full game simulation, running config.num_simulations per iteration,
        choosing nodes either deterministically (explore=False) or via softmax sampling
        (explore=True) for subsequent iterations.

        Called recursively, returning a generator of game positions:
        >>> game = list(start.run_mcts(explore=True))
        """
        
        logger.info(
            f"{self._game.id}: selecting node {self.graph_node} with value={self.value:.3f} and visits={self.visits}")
        
        yield self
        
        if not self.terminal:
            for _ in range(num_simulations):
                self.mcts_step()
            
            # Store the search statistics and policy inputs for network training
            self.store_policy_data()
            
            # select action -- either softmax sample or by visit count
            if explore:
                choice = self.softmax_sample()
            
            else:
                choice = sorted((node for node in self.get_successors()),
                                key=lambda x: -x.visits)[0]
            
            yield from choice.run_mcts(num_simulations, explore=explore)
