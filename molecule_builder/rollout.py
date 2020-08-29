"""AlphaZero rollout worker script."""

from functools import lru_cache

import numpy as np
import pickle

from rdkit import Chem
import rdkit.Chem.AllChem
from rdkit.Chem.Descriptors import qed

from molecule_builder import build_molecules

from config import AlphaZeroConfig
from network import Network
from training import sample, model_training

CONFIG = AlphaZeroConfig()


# Create cached functions
@lru_cache(maxsize=CONFIG.lru_cache_maxsize)
def get_mol_from_smiles(smiles):
    """Returns the RDKIT mol given the SMILES string"""
    return Chem.MolFromSmiles(smiles)

@lru_cache(maxsize=CONFIG.lru_cache_maxsize)
def get_smiles_from_mol(mol):
    """Returns the SMILES string given the RDKIT mol"""
    return Chem.MolToSmiles(mol)

@lru_cache(maxsize=CONFIG.lru_cache_maxsize)
def get_reward_from_mol(mol):
    """Returns the reward."""
    return qed(mol)

@lru_cache(maxsize=CONFIG.lru_cache_maxsize)
def get_next_mols(mol, fp_length):
    """Returns list of next SMILES strings and fingerprints."""
    mols = [m for m in build_molecules(mol)]
    mols_fp = [get_fingerprint(m, fp_length=fp_length) for m in mols]
    return [Chem.MolToSmiles(m) for m in mols], mols_fp

@lru_cache(maxsize=CONFIG.lru_cache_maxsize)
def get_fingerprint(mol, radius=2, fp_length=128):
    """Returns the Morgan fingerprint for given RDKIT mol."""
    fingerprint = rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius, fp_length)
    arr = np.zeros((fp_length,), dtype=np.float32)
    rdkit.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


class Node(object):
    """TODO: use networkx to give unique hashes to unique nodes."""

    def __init__(self, prior: float, mol: str):
        self.visit_count = 0
        self.prior = prior
        self.mol = mol
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Game(object):

    def __init__(self, history=None):

        self.history = history or ["C"]
        self.child_visits = []
        self.mol = get_mol_from_smiles(self.history[-1])

        self.max_actions = CONFIG.max_next_mols
        self.fingerprint_dim = CONFIG.fingerprint_dim
        self.max_atoms = CONFIG.max_atoms
        self.action_mask = np.zeros(CONFIG.max_next_mols, dtype=np.float32)

        self.root_next_mols_fp = []
        self.next_mols, self.next_mols_fp = get_next_mols(
            self.mol, fp_length=self.fingerprint_dim)

    def terminal(self):
        return self.mol.GetNumAtoms() == self.max_atoms or len(self.next_mols) == 0

    def terminal_value(self, state_index):
        mol = get_mol_from_smiles(self.history[-1])
        return get_reward_from_mol(mol)
    
    def history_fps(self):
        mols = [get_mol_from_smiles(self.history[i]) for i in range(len(self.history))]
        return [get_fingerprint(mol) for mol in mols]
    
    def root_next_mols(self):
        for i in range(len(self.history)):
            _, next_mols_fp = get_next_mols(get_mol_from_smiles(self.history[i]), fp_length=self.fingerprint_dim)
            self.root_next_mols_fp.append(next_mols_fp)
        return self.root_next_mols_fp
    
    def get_action_mask(self):
        self.action_mask[:len(self.next_mols)] = 1
        return self.action_mask

    def legal_actions(self):
        return range(len(self.next_mols))

    def clone(self):
        return Game(list(self.history))

    def apply(self, action):
        self.mol = self.next_mols[action]
        self.history.append(self.mol)
        self.mol = get_mol_from_smiles(self.mol)
        if self.mol.GetNumAtoms() == self.max_atoms:
            self.next_mols, self.next_mols_fp = [], []
        else:
            self.next_mols, self.next_mols_fp = get_next_mols(
                self.mol, fp_length=self.fingerprint_dim)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.max_actions)
        ])

    def make_inputs(self, state_index):
        mol = get_mol_from_smiles(self.history[state_index])
        _, mols = get_next_mols(mol, fp_length=self.fingerprint_dim)
        mol = get_fingerprint(mol, fp_length=self.fingerprint_dim)
        next_mols = np.zeros((self.max_actions, self.fingerprint_dim), dtype=np.float32)
        action_mask = np.zeros(self.max_actions, dtype=np.float32)
        if len(mols) > 0:
            next_mols[:len(mols), :] = mols
            action_mask[:len(mols)] = 1.
        return mol, next_mols, action_mask

    
def save_game(game, i, args):
    """Push the game to the buffer.  Suggested data structure (dict):
        data = {
            "network_inputs": {
                "mol":  fingerprints each molecule in game history,
                "next_mols": fingerprint arrays of chilren molecules for
                    each root molecule,
                "pi":  child visit arrays for each root molecule,
            }
            "mol_smiles": smiles strings for game history (list),
            "reward": terminal reward (float)
        }

    Erotokritos, you might want to modify the Game class to keep track of these
    things when they are generated during MCTS, so you don't have to recreate
    them here.
    """

    data = {
            "network_inputs": {
                "mol":  [game.make_inputs(i)[0] for i in range(len(game.history)-1)],
                "next_mols": [game.make_inputs(i)[1] for i in range(len(game.history)-1)],
                "action_mask": [game.make_inputs(i)[2] for i in range(len(game.history)-1)],
                "pi":  [game.child_visits[i] for i in range(len(game.history)-1)],
            },
            "mol_smiles": game.history,
            "reward": game.terminal_value(-1)
        }
   
    with open('game_{}_{}.pickle'.format(i, args.id), 'wb') as f:
        pickle.dump(data, f)


def play_game(network, explore=True):
    game = Game()
    game_step = 0
    while not game.terminal() and len(game.history) < game.max_atoms:
        action, root = run_mcts(game, network, explore)
        game.apply(action)
        game.store_search_statistics(root)
        game_step += 1
    return game


def run_mcts(game, network, explore=True):
  
    root = Node(0, game.history[0])
    evaluate(root, game, network)

    # # Do we want to use exploration here, or only at action selection?
    # if explore:
    #   add_exploration_noise(config, root)

    for _ in range(CONFIG.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        mcts_step = 0
        while node.expanded():
            action, node = select_child(node)
            scratch_game.apply(action)
            search_path.append(node)
            mcts_step += 1

        if scratch_game.terminal():
            value = scratch_game.terminal_value(-1)
        else:
            value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value)
  
    return select_action(game, root, explore), root


def select_action(game, root, explore=True):
    """Choose the action via either softmax or max visit count."""
    visit_counts = [(child.visit_count, action)
                      for action, child in root.children.items()]
    if explore:
        action = softmax_sample(visit_counts)
    else:
        action = np.argmax([x[0] for x in visit_counts])
    return action


def select_child(node):
    """Select the child with the highest UCB score."""
    _, action, child = max((ucb_score(node, child), action, child)
                            for action, child in node.children.items())
    return action, child


def ucb_score(parent, child):
    """The score for a node is based on its value, plus an exploration bonus based on
    the prior."""
    pb_c = np.log((parent.visit_count + CONFIG.pb_c_base + 1) /
                    CONFIG.pb_c_base) + CONFIG.pb_c_init
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


def evaluate(node, game, network):
    """Call the neural network to obtain a value and policy prediction."""

    mol, next_mols, action_mask = game.make_inputs(-1)
    value, policy_logits = network.inference(mol, next_mols, action_mask)   

    # Expand the node.
    policy = {a: np.exp(float(policy_logits[a])) for a in game.legal_actions()}
    policy_sum = float(sum(policy.values()))
    for ix, (action, p) in enumerate(policy.items()):
        node.children[action] = Node(p/policy_sum, game.next_mols[ix])
    return value


def backpropagate(search_path, value):
    """At the end of a simulation, we propagate the evaluation all the way up the
    tree to the root."""
    for node in search_path:
        node.value_sum += value
        node.visit_count += 1


def add_exploration_noise(node):
    """At the start of each search, we add dirichlet noise to the prior of the root
    to encourage the search to explore new actions."""
    actions = node.children.keys()
    noise = np.random.gamma(CONFIG.root_dirichlet_alpha, 1, len(actions))
    frac = CONFIG.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def softmax_sample(d):
    """Perform softmax sampling for a vector d."""
    d = np.exp([x[0] for x in d])
    d_sum = d.sum()
    return np.random.choice(range(len(d)), size=1, p=d/d_sum)[0]


def rollout_loop(args):
    """Main rollout loop that plays games using the latest network weights,
    and pushes games to the replay buffer."""
    network = Network(args.checkpoint_dir)
    pckl_list = []
    for i in range(CONFIG.num_rollouts):
        print("updating network weights")
        network.load_weights()
        print("playing game")
        game = play_game(network)
        print("saving game")
        save_game(game, i, args)
    network.compile()
    model_training(network, args)

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=None,
        help="Directory containing saved network checkpoints")
    parser.add_argument("--id", type=int, default=None, help="worker id")
    args = parser.parse_args()
    print(args)

    rollout_loop()

