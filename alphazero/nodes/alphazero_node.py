import io
import itertools
from abc import abstractmethod
from typing import (
    Iterable,
    List,
    Optional,
    )

from keras_preprocessing.sequence import pad_sequences

from alphazero.nodes.graph_node import GraphNode
import numpy as np
import tensorflow as tf


class AlphaZeroNode(GraphNode):
    
    def __init__(self) -> None:
        self.visits: int = 0
        self.total_value: float = 0.0
        self.prior_logit: float = np.nan
        self._reward: Optional[float] = None
        self._child_priors: Optional[List['AlphaZeroNode']] = None
        self._policy_inputs: Optional[{}] = None
    
    @abstractmethod
    @property
    def successors(self) -> Iterable['AlphaZeroNode']:
        pass
    
    @abstractmethod
    def compute_reward(self) -> float:
        pass
    
    def update(self, reward: float) -> None:
        self.visits += 1
        self.total_value += reward
    
    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits != 0 else 0
    
    def ucb_score(self, parent):
        pb_c = np.log((parent.visits + config.pb_c_base + 1) /
                      config.pb_c_base) + config.pb_c_init
        
        pb_c *= np.sqrt(parent.visits) / (self.visits + 1)
        
        prior_score = pb_c * self.prior(parent)
        
        return prior_score + self.value
    
    def reset_priors(self) -> None:
        self.prior_logit = np.nan
    
    def reset_updates(self) -> None:
        self.visits = 0
        self.total_value = 0
        self._reward = None
    
    def prior(self, parent) -> float:
        """Prior probabilities (unlike logits) depend on the parent"""
        return parent.child_priors[self]
    
    @property
    def child_priors(self) -> {'AlphaZeroNode': float}:
        """Get a list of priors for the node's children, with optionally added dirichlet noise.
        Caches the result, so the noise is consistent.
        """
        if self._child_priors is None:
            # Perform the softmax over the children node's prior logits
            children = list(self.successors)
            priors = tf.nn.softmax([child.prior_logit for child in children]).numpy()
            
            # Add the optional exploration noise
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
        """Constructs GNN inputs for the node, or returns them if they've
        been previously cached"""
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
        visit_counts = np.array([child.visits for child in self.successors])
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
