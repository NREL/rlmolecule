import logging
import os
import uuid
from pprint import pprint
from typing import Optional

import networkx as nx
import numpy as np
import tensorflow as tf

from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromSmiles

import alphazero.config as config
from alphazero.node import Node
from alphazero.nodes.alpha_zero_game import AlphaZeroGame
from alphazero.policy import build_policy_trainer
from alphazero.preprocessor import MolPreprocessor
from molecule_graph.molecule_node import MoleculeNode

logger = logging.getLogger(__name__)

default_preprocessor = MolPreprocessor(atom_features=atom_featurizer,
                                       bond_features=bond_featurizer,
                                       explicit_hs=False)

default_preprocessor.from_json(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'preprocessor.json'))


class MoleculeGame(AlphaZeroGame):
    
    def __init__(self, config: any, start_smiles: str, preprocessor: MolPreprocessor = default_preprocessor) -> None:
        super().__init__(config)
        self.preprocessor: MolPreprocessor = preprocessor
        self.start: MoleculeNode = MoleculeNode.make_from_SMILES(self, MolFromSmiles(start_smiles))
        pprint(self.start)
        
        policy_trainer = build_policy_trainer()
        policy_model = policy_trainer.layers[-1].policy_model
        policy_predictions = tf.function(experimental_relax_shapes=True)(policy_model.predict_step)
        self._setup(policy_trainer, policy_model, policy_predictions)
    
    def expand(self, parent: MoleculeNode):
        """For a given node, build the chidren, add them to the graph, and run the
        policy network to get prior_logits and a value.
        
        Returns:
        value (float): the estimated value of `parent`.
        """
        
        # Create the children nodes and add them to policy_predictionsthe graph
        children = list(parent.successors)  # TODO: integration point
        
        # Handle the case where a node doesn't have any valid children
        if not children:
            parent.terminal = True
            parent._reward = config.min_reward
            return parent._reward
        
        # Run the policy network to get value and prior_logit predictions
        values, prior_logits = self.policy_predictions(
            parent.policy_inputs_with_children())  # TODO: integration point - policy_inputs_with_children generates
        # inputs to policy network
        prior_logits = prior_logits[1:].numpy().flatten()
        
        # Update child nodes with predicted prior_logits
        for child, prior_logit in zip(parent.successors, prior_logits):
            child._prior_logit = float(prior_logit)
        
        # Return the parent's predicted value
        return float(tf.nn.sigmoid(values[0]))
    
    def mcts_step(self, start: MoleculeNode):
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
    def softmax_sample(node: MoleculeNode) -> MoleculeNode:
        """Sample from node.successors according to their visit counts.
        
        Returns:
            choice: Node, the chosen successor node.
        """
        successors = list(node.successors)
        visit_counts = np.array([n._visits for n in successors])
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
                                key=lambda x: x._visits, reverse=True)[0]
            
            yield from self.run_mcts(choice, explore=explore)
