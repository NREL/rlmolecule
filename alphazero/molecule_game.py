import logging
import uuid
from pprint import pprint
from typing import Optional

import networkx as nx
import numpy as np
import tensorflow as tf

from rdkit import Chem

import alphazero.config as config
from alphazero.node import Node
from alphazero.policy import build_policy_trainer
from molecule_graph.molecule_node import MoleculeNode

logger = logging.getLogger(__name__)


class MoleculeGame:
    
    def __init__(self, config: any, start_smiles: str):
        self._config: any = config
        self.id = uuid.uuid4().hex[:8]
        self.start = MoleculeNode.make_from_SMILES(self, start_smiles)
        
        pprint(self.start)
        
        self.start = start
        
        self.policy_trainer = build_policy_trainer()  # TODO: integration point - policy network must match node impl
        # (node generates inputs to network)
        self.policy_model = self.policy_trainer.layers[-1].policy_model
        
        latest = tf.train.latest_checkpoint(config.checkpoint_filepath)
        if latest:
            self.policy_trainer.load_weights(latest)
            logger.info(f'{self.id}: loaded checkpoint {latest}')
        else:
            logger.info(f'{self.id}: no checkpoint found')
        
        self.policy_predictions = tf.function(experimental_relax_shapes=True)(
            self.policy_model.predict_step)
    
    @property
    def config(self) -> any:
        return self._config
    
    def tree_policy(self, parent: Node):
        """Implements the tree search part of an MCTS search. Recursive function which
        returns a generator over the optimal path.
    
        >>> history = list(tree_policy(G, start))"""
        
        yield parent
        
        if not parent.terminal:
            
            sorted_successors = sorted(
                parent.successors, key=lambda x: x.ucb_score(parent), reverse=True)
            
            if sorted_successors:
                yield from self.tree_policy(sorted_successors[0])
    
    def expand(self, parent: Node):
        """For a given node, build the chidren, add them to the graph, and run the
        policy network to get prior_logits and a value.
        
        Returns:
        value (float): the estimated value of `parent`.
        """
        
        # Create the children nodes and add them to the graph
        children = list(parent.build_children())  # TODO: integration point
        
        # Handle the case where a node doesn't have any valid children
        if not children:
            parent.terminal = True
            parent._reward = config.min_reward
            return parent._reward
        
        self.add_edges_from(((parent, child) for child in
                             children))  # TODO: networkx call - uses hash function to figure out if a new node is
        # needed
        
        # Run the policy network to get value and prior_logit predictions
        values, prior_logits = self.policy_predictions(
            parent.policy_inputs_with_children())  # TODO: integration point - policy_inputs_with_children generates
        # inputs to policy network
        prior_logits = prior_logits[1:].numpy().flatten()
        
        # Update child nodes with predicted prior_logits
        for child, prior_logit in zip(parent.successors, prior_logits):
            child.prior_logit = float(prior_logit)
        
        # Return the parent's predicted value
        return float(tf.nn.sigmoid(values[0]))
    
    def mcts_step(self, start: Node):
        """Perform a single MCTS step from the given starting node, including a
        tree search, expansion, and backpropagation.
        """
        
        # Perform the tree policy search
        history = list(self.tree_policy(start))
        leaf = history[-1]
        
        # Looks like in alphazero, we always expand, even if this is the
        # first time we've visited the node
        if not leaf.terminal:
            value = self.expand(leaf)
        else:
            value = leaf.reward
        
        # perform backprop
        for node in history:
            node.update(value)
        
        return leaf
    
    @staticmethod
    def softmax_sample(node: Node) -> Node:
        """Sample from node.successors according to their visit counts.
        
        Returns:
            choice: Node, the chosen successor node.
        """
        successors = list(node.successors)
        visit_counts = np.array([n.visits for n in successors])
        visit_softmax = tf.nn.softmax(tf.constant(visit_counts, dtype=tf.float32)).numpy()
        return successors[np.random.choice(
            range(len(successors)), size=1, p=visit_softmax)[0]]
    
    def run_mcts(self, node: Node = None, explore: bool = True):
        """Performs a full game simulation, running config.num_simulations per iteration,
        choosing nodes either deterministically (explore=False) or via softmax sampling
        (explore=True) for subsequent iterations.
        
        Called recursively, returning a generator of game positions:
        >>> game = list(run_mcts(G, start, explore=True))
        """
        
        # The starting node of the current MCTS iteration.
        node = node if node else self.start
        
        logger.info(f"{self.id}: selecting node {node} with value={node.value:.3f} and visits={node.visits}")
        yield node
        
        if not node.terminal:
            for _ in range(config.num_simulations):
                self.mcts_step(node)
            
            # Store the search statistics and policy inputs for network training
            node.store_policy_inputs_and_targets()
            
            # select action -- either softmax sample or by visit count
            if explore:
                choice = self.softmax_sample(node)
            
            else:
                choice = sorted((node for node in start.successors),
                                key=lambda x: x.visits, reverse=True)[0]
            
            yield from self.run_mcts(choice, explore=explore)
