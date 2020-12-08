import io
import itertools
from abc import abstractmethod
from typing import (
    Iterable,
    Iterator,
    List,
    Optional,
    )

from keras_preprocessing.sequence import pad_sequences

from alphazero.nodes.graph_node import GraphNode
import numpy as np
import tensorflow as tf


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
    
    def get_successors(self) -> Iterable['AlphaZeroNode']:
        game = self._game
        return (AlphaZeroNode(graph_successor, game) for graph_successor in self._graph_node.get_successors())
    
    def compute_reward(self) -> float:
        # if self.terminal:
        #     return config.min_reward
        # TODO: more here
        pass
    
    def compute_value_estimate(self) -> float:
        """
        For a given node, build the chidren, add them to the graph, and run the
        policy network to get prior_logits and a value.

        Returns:
        value (float): the estimated value of `parent`.
        """
        
        children = list(self.get_successors())
        
        # Handle the case where a node doesn't have any valid children
        if len(children) == 0:
            return self.config.min_reward
        
        # Run the policy network to get value and prior_logit predictions
        # TODO: integration point - policy_predictions
        values, prior_logits = self.policy_predictions(self.policy_inputs_with_children())
        
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
    def config(self) -> any:
        return self._game.config
    
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
        yield from sorted(self.get_successors(), key=lambda successor: -self.ucb_score(successor))
    
    def ucb_score(self, child: 'AlphaZeroNode') -> float:
        config = self.config
        pb_c = np.log((self.visits + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
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
            children = list(self.get_successors())
            priors = tf.nn.softmax([child._prior_logit for child in children]).numpy()
            
            # Add the optional exploration noise
            config = self.config
            if config.dirichlet_noise:
                random_state = np.random.RandomState()
                noise = random_state.dirichlet(
                    np.ones_like(priors) * config.dirichlet_alpha)
                
                priors = priors * (1 - config.dirichlet_x) + (noise * config.dirichlet_x)
            
            assert np.isclose(priors.sum(), 1.)  # Just a sanity check
            self._child_priors = {child: prior for child, prior in zip(children, priors)}
        
        return self._child_priors
    
    # TODO: integration point - uses preprocessor script to build inputs to policy network
    @property
    def policy_inputs(self):
        """
        Constructs GNN inputs for the node, or returns them if they've been previously cached
        """
        if self._policy_inputs is None:
            self._policy_inputs = preprocessor.construct_feature_matrices(self)
        return self._policy_inputs
    
    # TODO: integration point - stacks parent with current children as a batch
    def policy_inputs_with_children(self) -> {}:
        """Return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network"""
        
        policy_inputs = [node.policy_inputs for node in itertools.chain((self,), self.successors)]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}
    
    # TODO: integration point - stores inputs so it can go into database
    def store_policy_inputs_and_targets(self) -> None:
        """Stores the output of `self.policy_inputs_with_children` and child visit probabilities
        as a numpy compressed binary data string.

        save and load using the following:
        >>> self.store_policy_inputs_and_targets()
        >>> with io.BytesIO(self._policy_data) as f:
            data = dict(np.load(f, allow_pickle=True).items())

        """
        
        data = self.policy_inputs_with_children()
        visit_counts = np.array([child._visits for child in self.successors])
        data['visit_probs'] = visit_counts / visit_counts.sum()
        
        with io.BytesIO() as f:
            np.savez_compressed(f, **data)
            binary_data = f.getvalue()
        
        self._policy_data = binary_data
    
    @property
    def reward(self) -> float:
        assert self.terminal, "Accessing reward of non-terminal state"
        if self._reward is None:
            self._reward = self.compute_reward()
        return self._reward
