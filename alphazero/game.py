import networkx as nx
import numpy as np
import tensorflow as tf

from rdkit import Chem

import alphazero.config as config
from alphazero.node import Node
from alphazero.policy import policy_model

model = policy_model()

class Game(nx.DiGraph):

    def __init__(self, node_cls=None, start_smiles=None, checkpoint_dir=None):
        super(Game, self).__init__()

        start = node_cls(Chem.MolFromSmiles(start_smiles), graph=self)
        self.add_node(start)
        self.start = start

        self.checkpoint_dir = checkpoint_dir
        self.tf_checkpoint = None
        self.reset()
        
        self.dirichlet_noise = True
        self.dirichlet_alpha = 1.
        self.dirichlet_d = 0.25


    def reset(self):

        # Check for new weights.  If they exist, update the model and clear out
        # all priors on the graph.
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest and self.tf_checkpoint != latest:

            # Update the latest checkpoint
            self.tf_checkpoint = latest
            model.load_weights(latest)

            # Clear out node priors
            _ = [node.reset_priors() for node in self]

        # And no matter what, clear out visit and value data on the graph
        _ = [node.reset_updates() for node in self]
        

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
        self.add_edges_from(((parent, child) for child in parent.build_children()))
        
        # Run the policy network to get value and prior_logit predictions
        values, prior_logits = model(parent.policy_inputs_with_children())
        prior_logits = prior_logits[1:].numpy().flatten()
        
        # if we're adding noise, perturb the logits
        if self.dirichlet_noise:
            random_state = np.random.RandomState()
            noise = random_state.dirichlet(
                np.ones_like(prior_logits) * self.dirichlet_alpha)
            prior_logits += np.exp(prior_logits).sum() * noise * self.dirichlet_d
        
        # Update child nodes with predicted prior_logits
        for child, prior_logit in zip(parent.successors, prior_logits):
            child.prior_logit = float(prior_logit)
            
        # Return the parent's predicted value
        return float(values[0])


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
        
    
    def run_mcts(self, node: Node=None, explore: bool=True):
        """Performs a full game simulation, running config.num_simulations per iteration,
        choosing nodes either deterministically (explore=False) or via softmax sampling
        (explore=True) for subsequent iterations.
        
        Called recursively, returning a generator of game positions:
        >>> game = list(run_mcts(G, start, explore=True))
        """

        node = node if node else self.start
        yield node

        if not node.terminal:
            for _ in range(config.num_simulations):
                self.mcts_step(node)
            
            # select action -- either softmax sample or by visit count
            if explore:
                choice = self.softmax_sample(node)
                
            else:
                choice = sorted((node for node in start.successors), 
                                key=lambda x: x.visits, reverse=True)[0]
                
            yield from self.run_mcts(choice, explore=explore)
