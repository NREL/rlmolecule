from abc import (
    ABC,
    abstractmethod,
    )
from typing import Iterable

import numpy as np


class AbstractNode(ABC):
    
    @property
    def terminal(self) -> bool:
        return not any(True for _ in self.successors)
    
    @abstractmethod
    @property
    def successors(self) -> Iterable['AbstractNode']:
        pass
    
    @abstractmethod
    @property
    def reward(self) -> float:
        pass
    
    # @abstractmethod
    # def update(self, reward: float) -> None:
    #     pass
    #
    # @abstractmethod
    # @property
    # def visits(self) -> int:
    #     pass
    #
    # @abstractmethod
    # @property
    # def value(self) -> float:
    #     pass
    
    # def ucb_score(self, parent):
    #     # gets results for this node from the networkx digraph
    #     pb_c = np.log((parent.visits + config.pb_c_base + 1) /
    #                   config.pb_c_base) + config.pb_c_init
    #
    #     pb_c *= np.sqrt(parent.visits) / (self.visits + 1)
    #
    #     prior_score = pb_c * self.prior(parent)
    #
    #     return prior_score + self.value
    
    # @property
    # def prior_logit(self):
    #     node = self.game.nodes[self]
    #     try:
    #         return node['prior_logit']
    #     except KeyError:
    #         return np.nan
    #
    # def reset_priors(self):
    #     node = self.game.nodes[self]
    #     if 'prior_logit' in node:
    #         del node['prior_logit']
    #
    # def reset_updates(self):
    #     node = self.game.nodes[self]
    #     if 'visits' in node:
    #         del node['visits']
    #
    #     if 'total_value' in node:
    #         del node['total_value']
    #
    #     if hasattr(self, '_reward'):
    #         del self._reward
    #
    # @prior_logit.setter
    # def prior_logit(self, value):
    #     node = self.game.nodes[self]
    #
    #     if 'prior_logit' in node:
    #         pass
    #
    #     else:
    #         node['prior_logit'] = value
    
    # def prior(self, parent):
    #     """Prior probabilities (unlike logits) depend on the parent"""
    #     return parent.child_priors[self]
    
    # @property
    # def policy_inputs(self):
    #     """Constructs GNN inputs for the node, or returns them if they've
    #     been previously cached"""
    #     # TODO: integration point - uses preprocessor script to build inputs to policy network
    #     try:
    #         return self._policy_inputs
    #     except AttributeError:
    #         self._policy_inputs = preprocessor.construct_feature_matrices(self)
    #         return self._policy_inputs
    #
    # def policy_inputs_with_children(self):
    #     """Return the given nodes policy inputs, concatenated together with the
    #     inputs of its successor nodes. Used as the inputs for the policy neural
    #     network"""
    #     # TODO: integration point - stacks parent with current children as a batch
    #     policy_inputs = [node.policy_inputs for node in itertools.chain((self,), self.successors)]
    #     return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
    #             for key in policy_inputs[0].keys()}
    #
    # def get_action_inputs_as_binary(self):
    #     """Returns the output of `self.policy_inputs_with_children` and child visit probabilities
    #     as a numpy compressed binary data string.
    #
    #     save and load using the following:
    #     >>> binary_data = self.get_action_inputs_as_binary()
    #     >>> with io.BytesIO(binary_data) as f:
    #         data = dict(np.load(f, allow_pickle=True).items())
    #
    #     """
    #     # TODO: integration point - stores inputs so it can go into database
    #     data = self.policy_inputs_with_children()
    #     visit_counts = np.array([child.visits for child in self.successors])
    #     data['visit_probs'] = visit_counts / visit_counts.sum()
    #
    #     with io.BytesIO() as f:
    #         np.savez_compressed(f, **data)
    #         binary_data = f.getvalue()
    #     return binary_data
    #
    # @property
    # def child_priors(self):
    #     """Get a list of priors for the node's children, with optionally added dirichlet noise.
    #     Caches the result, so the noise is consistent.
    #     """
    #     try:
    #         return self._child_priors
    #
    #     except AttributeError:
    #
    #         # Perform the softmax over the children node's prior logits
    #         children = list(self.successors)
    #         priors = tf.nn.softmax([child.prior_logit for child in children]).numpy()
    #
    #         # Add the optional exploration noise
    #         if config.dirichlet_noise:
    #             random_state = np.random.RandomState()
    #             noise = random_state.dirichlet(
    #                 np.ones_like(priors) * config.dirichlet_alpha)
    #
    #             priors = priors * (1 - config.dirichlet_x) + (noise * config.dirichlet_x)
    #
    #         assert np.isclose(priors.sum(), 1.)  # Just a sanity check
    #         self._child_priors = {child: prior for child, prior in zip(children, priors)}
    #         return self._child_priors
