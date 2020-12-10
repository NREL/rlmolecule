import io
import itertools

import networkx as nx
import numpy as np
import rdkit
import rdkit.Chem
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import molecule_game.config as config
from molecule_game.molecule_tools import build_molecules, build_radicals
from molecule_game.mol_preprocessor import preprocessor

# TODO: can you remove rdkit from /alphazero/ ?

class Node(rdkit.Chem.Mol): # TODO: integration point - factor into implemenation
    
    def __init__(self, *args, graph: nx.DiGraph = None, terminal: bool = False, **kwargs):
        """Base class that handles much of the mcts alphazero logic for
        organic radical construction. Actual usage of this class requires
        subclassing with a specific reward function, i.e.

        >>> from rdkit.Chem.Descriptors import qed

        >>> class QedNode(Node):
                def get_reward(self):
                    return qed(self)

        >>> G = nx.DiGraph()
        >>> start = QedNode(rdkit.Chem.MolFromSmiles('CC'), graph=G)
        >>> G.add_node(start)
        >>> game = list(run_mcts(G, start))

        Parameters:
        *args: standard initialization of rdkit.Chem.Mol using another Mol
        graph: A networkx.DiGraph definining the MCTS tree
        terminal: whether the given node is a terminal state

        """
        
        super(Node, self).__init__(*args, **kwargs)
        self.G = graph
        self.terminal = terminal
        
    def __hash__(self):
        return hash(self.smiles) # TODO: integration point

    def __eq__(self, other):
        return self.__hash__() == other.__hash__() # TODO: integration point (w/network x)
    
    @property
    def smiles(self): # TODO: integration point - factor into implementation
        return rdkit.Chem.MolToSmiles(self)
    
    def __repr__(self):
        return '<{}>'.format(self.smiles)
    
    # TODO: this calls the molecule builder -- factor into implementation
    def build_children(self):
        if self.terminal:
            raise RuntimeError("Attemping to get children of terminal node")
    
        if self.GetNumAtoms() < config.max_atoms:
            for mol in build_molecules(self, **config.build_kwargs):
                if self.G.has_node(mol):
                    # Check if the graph already has the current mol
                    yield self.G.nodes[mol]
                else:
                    yield self.__class__(mol, graph=self.G)
    
        if self.GetNumAtoms() >= config.min_atoms:
            for radical in build_radicals(self):
                yield self.__class__(radical, graph=self.G, terminal=True)

    @property
    def successors(self):
        return self.G.successors(self)

    def update(self, reward):
        """ Value and visit information is stored in the networkx graph, not individual nodes. """
        node = self.G.nodes[self]
        if 'visits' in node:
            node['visits'] += 1
        else:
            node['visits'] = 1
    
        if 'total_value' in node:
            node['total_value'] += reward
        else:
            node['total_value'] = reward

    @property
    def visits(self):
        node = self.G.nodes[self]
        try:
            return node['visits']
        except KeyError:
            return 0

    @property
    def value(self):
        node = self.G.nodes[self]
    
        try:
            total_value = node['total_value']
        except KeyError:
            total_value = 0
    
        return total_value / self.visits if self.visits > 0 else 0

    def ucb_score(self, parent):
    
        pb_c = np.log((parent.visits + config.pb_c_base + 1) /
                      config.pb_c_base) + config.pb_c_init
    
        pb_c *= np.sqrt(parent.visits) / (self.visits + 1)
    
        prior_score = pb_c * self.prior(parent)
    
        return prior_score + self.value

    @property
    def prior_logit(self):
        node = self.G.nodes[self]
        try:
            return node['prior_logit']
        except KeyError:
            return np.nan

    def reset_priors(self):
        node = self.G.nodes[self]
        if 'prior_logit' in node:
            del node['prior_logit']

    def reset_updates(self):
        node = self.G.nodes[self]
        if 'visits' in node:
            del node['visits']
    
        if 'total_value' in node:
            del node['total_value']
    
        if hasattr(self, '_reward'):
            del self._reward

    @prior_logit.setter
    def prior_logit(self, value):
        node = self.G.nodes[self]
    
        if 'prior_logit' in node:
            pass
    
        else:
            node['prior_logit'] = value

    def prior(self, parent):
        """Prior probabilities (unlike logits) depend on the parent"""
        return parent.child_priors[self]

    # TODO: integration point - uses preprocessor script to build inputs to policy network
    @property
    def policy_inputs(self):
        """Constructs GNN inputs for the node, or returns them if they've
        been previously cached"""
        try:
            return self._policy_inputs
        except AttributeError:
            self._policy_inputs = preprocessor.construct_feature_matrices(self)
            return self._policy_inputs

    # TODO: integration point - stacks parent with current children as a batch
    def policy_inputs_with_children(self):
        """Return the given nodes policy inputs, concatenated together with the 
        inputs of its successor nodes. Used as the inputs for the policy neural
        network"""
    
        policy_inputs = [node.policy_inputs for node in itertools.chain((self,), self.successors)]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}

    # TODO: integration point - stores inputs so it can go into database
    def store_policy_inputs_and_targets(self):
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
    def child_priors(self):
        """Get a list of priors for the node's children, with optionally added dirichlet noise.
        Caches the result, so the noise is consistent.
        """
        try:
            return self._child_priors
    
        except AttributeError:
        
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

    # TODO: integration point -
    @property
    def reward(self):
        assert self.terminal, "Accessing reward of non-terminal state"
        try:
            return self._reward
        except AttributeError:
            self._reward = self.get_reward()
            return self._reward

    def get_reward(self):
        """This should get overwritten by a subclass's reward function.
        (Should this be using ranked rewards?)
        """
        pass
