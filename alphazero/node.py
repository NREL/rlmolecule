import io
import itertools
from math import exp

import networkx as nx
import numpy as np
import rdkit.Chem
from tensorflow.keras.preprocessing.sequence import pad_sequences

from alphazero.config import AlphaZeroConfig
from alphazero.molecule import build_molecules, build_radicals
from alphazero.preprocessor import preprocessor

CONFIG = AlphaZeroConfig()

class Node(rdkit.Chem.Mol):
    
    def __init__(self, *args, graph: nx.DiGraph=None, terminal: bool=False, **kwargs):
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
        return hash(self.smiles)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
    
    @property
    def smiles(self):
        return rdkit.Chem.MolToSmiles(self)
    
    def __repr__(self):
        return '<{}>'.format(self.smiles)
    
    def build_children(self):
        if self.terminal:
            raise RuntimeError("Attemping to get children of terminal node")
        
        if self.GetNumAtoms() < CONFIG.max_atoms:
            for mol in build_molecules(self, stereoisomers=False):
                if self.G.has_node(mol):
                    # Check if the graph already has the current mol
                    yield self.G.nodes[mol]
                else:
                    yield self.__class__(mol, graph=self.G)
        
        if self.GetNumAtoms() >= CONFIG.min_atoms:
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
        
        pb_c = np.log((parent.visits + CONFIG.pb_c_base + 1) /
                      CONFIG.pb_c_base) + CONFIG.pb_c_init
        
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
    
    @prior_logit.setter
    def prior_logit(self, value):
        node = self.G.nodes[self]
        
        if 'prior_logit' in node:
            pass
        
        else:
            node['prior_logit'] = value

    def prior(self, parent):
        """Prior probabilities (unlike logits) depend on the parent"""
        return (exp(self.prior_logit) / 
                sum((exp(child.prior_logit) for child in parent.successors)))
            
    @property
    def policy_inputs(self):
        """Constructs GNN inputs for the node, or returns them if they've
        been previously cached"""
        try:
            return self._policy_inputs
        except AttributeError:
            self._policy_inputs = preprocessor.construct_feature_matrices(self)
            return self._policy_inputs

        
    def policy_inputs_with_children(self):
        """Return the given nodes policy inputs, concatenated together with the 
        inputs of its successor nodes. Used as the inputs for the policy neural
        network"""
        
        policy_inputs = [node.policy_inputs for node in itertools.chain((self,), self.successors)]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}

    
    def get_action_inputs_as_binary(self):
        """Returns the output of `self.policy_inputs_with_children` and child visit probabilities 
        as a numpy compressed binary data string.
        
        save and load using the following:
        >>> binary_data = self.get_action_inputs_as_binary()
        >>> with io.BytesIO(binary_data) as f:
            data = dict(np.load(f, allow_pickle=True).items())        
        
        """
        
        data = self.policy_inputs_with_children()
        visit_counts = np.array([child.visits for child in self.successors])
        data['visit_probs'] = visit_counts / visit_counts.sum()
        
        with io.BytesIO() as f:
            np.savez_compressed(f, **data)
            binary_data = f.getvalue()
        return binary_data
    
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