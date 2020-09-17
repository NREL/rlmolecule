import networkx as nx
import numpy as np
import tensorflow as tf

from alphazero.node import Node
from alphazero.policy import policy_model
from alphazero.config import AlphaZeroConfig

model = policy_model()
CONFIG = AlphaZeroConfig()

def tree_policy(G: nx.DiGraph, parent: Node):
    """Implements the tree search part of an MCTS search. Recursive function which
    returns a generator over the optimal path.
    
    >>> history = list(tree_policy(G, start))
    """

    yield parent

    if not parent.terminal:
        sorted_successors = sorted(
            parent.successors, key=lambda x: x.ucb_score(parent), reverse=True)
        
        if sorted_successors:
            yield from tree_policy(G, sorted_successors[0])


def expand(G: nx.DiGraph, parent: Node) -> float:
    """For a given node, build the chidren, add them to the graph, and run the 
    policy network to get prior_logits and a value.
    
    Returns:
    value (float): the estimated value of `parent`.
    """
    
    # Create the children nodes and add them to the graph
    G.add_edges_from(((parent, child) for child in parent.build_children()))
    
    # Run the policy network to get value and prior_logit predictions
    values, prior_logits = model(parent.policy_inputs_with_children())
    
    # Update child nodes with predicted prior_logits
    for child, prior_logit in zip(parent.successors, prior_logits[1:]):
        child.prior_logit = float(prior_logit)
        
    # Return the parent's predicted value
    return float(values[0])


def mcts_step(G: nx.DiGraph, start: Node):
    """Perform a single MCTS step from the given starting node, including a 
    tree search, expansion, and backpropagation.
    """
    
    # Perform the tree policy search
    history = list(tree_policy(G, start))
    leaf = history[-1]
    
    # Looks like in alphazero, we always expand, even if this is the 
    # first time we've visited the node
    if not leaf.terminal:
        value = expand(G, leaf)
    else:
        value = leaf.reward

    # perform backprop
    for node in history:
        node.update(value)
        
        
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
        
        
def run_mcts(G: nx.DiGraph, node: Node, explore: bool=True):
    """Performs a full game simulation, running CONFIG.num_simulations per iteration,
    choosing nodes either deterministically (explore=False) or via softmax sampling
    (explore=True) for subsequent iterations.
    
    Called recursively, returning a generator of game positions:
    >>> game = list(run_mcts(G, start, explore=True))
    """
    
    yield node

    if not node.terminal:
        for _ in range(CONFIG.num_simulations):
            mcts_step(G, node)
        
        # select action -- either softmax sample or by visit count
        if explore:
            choice = softmax_sample(node)
            
        else:
            choice = sorted((node for node in start.successors), 
                            key=lambda x: x.visits, reverse=True)[0]
            
        yield from run_mcts(G, choice, explore=explore)
        
