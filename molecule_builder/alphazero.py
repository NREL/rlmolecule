"""Pseudocode description of the AlphaZero algorithm."""

from __future__ import division

from collections import defaultdict
from copy import deepcopy
import math
import threading
import multiprocessing
import time
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Masking
from tensorflow.keras import Model
from tensorflow.keras import backend as K

from rdkit import Chem
import rdkit.Chem.AllChem
from rdkit.Chem.Descriptors import qed

from molecule_builder import build_molecules

##########################
####### Helpers ##########

BEST_MOLS = {2: "CO", 3: "C=CN", 4: "CCCO", 5: "C=Cn1[nH]o1"}

# Global caches to avoid repetitive calls to rdkit
NEXT_MOLS = {}
REWARDS = {}
SMILES_FROM_MOL = {}
MOL_FROM_SMILES = {}
FINGERPRINTS = {}

def get_mol_from_smiles(smiles):
  if smiles not in MOL_FROM_SMILES:
    MOL_FROM_SMILES[smiles] = Chem.MolFromSmiles(smiles)
  return MOL_FROM_SMILES[smiles]

def get_smiles_from_mol(mol):
  if mol not in SMILES_FROM_MOL:
    SMILES_FROM_MOL[mol] = Chem.MolToSmiles(mol)
  return SMILES_FROM_MOL[mol]

def get_reward_from_mol(mol):
  if mol not in REWARDS:
    REWARDS[mol] = qed(mol)
  return REWARDS[mol]

def get_next_mols(mol, fp_length):
  if mol not in NEXT_MOLS:
    mols = [m for m in build_molecules(mol, stereoisomers=False)]
    mols_fp = [get_fingerprint(m, fp_length=fp_length) for m in mols]
    NEXT_MOLS[mol] = {}
    NEXT_MOLS[mol][0] = [Chem.MolToSmiles(m) for m in mols]
    NEXT_MOLS[mol][1] = mols_fp
  return NEXT_MOLS[mol][0], NEXT_MOLS[mol][1]

def get_fingerprint(mol, radius=2, fp_length=128):
  if mol not in FINGERPRINTS:
    fingerprint = rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius, fp_length)
    arr = np.zeros((fp_length,), dtype=np.float32)
    rdkit.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    FINGERPRINTS[mol] = arr
  return FINGERPRINTS[mol]


class AlphaZeroConfig(object):

  def __init__(self):

    self.num_actors = 1   # uses background threading, so probably just keep at 1
    self.games_per_iteration = 16
    self.gradient_steps_per_update = 32

    self.r_alpha = 0.9
    self.max_atoms = 4  # max size of molecule
    self.num_simulations = 256  # number of simulations used by MCTS
    
    self.fingerprint_dim = 256     # molecule fingerprint dimension
    self.network_max_actions = 1024  # max number of possible next molecules
    self.network_num_units = 256    # number of hidden units in NN

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.03  # for chess, 0.03 for Go and 0.15 for shogi.
    # 0.3 is too high, at least for small molecules
    self.root_exploration_fraction = 0.25

    # UCB formula
    #self.pb_c_base = 19652
    self.pb_c_base = 1
    self.pb_c_init = 1.25

    ### Training
    self.training_steps = 1000 #int(700e3) # number of training batches used to udpate network
    self.evaluation_steps = 5  # how often to play an evaluation game with no exploration
    self.checkpoint_interval = 1 # int(128)  # how often to save the network to shared storage
    self.buffer_size = 256 #int(1e6)  # replay buffer size (number of games)
    self.batch_size = 256     # batch size used in training


class Node(object):

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

  def __init__(self, config, history=None):
    self.history = history or ["C"]
    self.config = config
    self.child_visits = []
    self.max_actions = self.config.network_max_actions
    self.fingerprint_dim = self.config.fingerprint_dim
    self.mol = get_mol_from_smiles(self.history[-1])
    self.next_mols, self.next_mols_fp = get_next_mols(self.mol, fp_length=self.fingerprint_dim)

  def terminal(self):
    return self.mol.GetNumAtoms() == self.config.max_atoms or len(self.next_mols) == 0

  def terminal_value(self, state_index):
    mol = get_mol_from_smiles(self.history[-1])
    return get_reward_from_mol(mol)
    
  def legal_actions(self):
    return range(len(self.next_mols))

  def clone(self):
    return Game(self.config, list(self.history))

  def apply(self, action):
    self.mol = self.next_mols[action]
    self.history.append(self.mol)
    self.mol = get_mol_from_smiles(self.mol)
    if self.mol.GetNumAtoms() == self.config.max_atoms:
      self.next_mols, self.next_mols_fp = [], []
    else:
      self.next_mols, self.next_mols_fp = get_next_mols(self.mol, fp_length=self.fingerprint_dim)

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.values())
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in range(self.max_actions)
    ])

  def make_inputs(self, state_index: int):
    mol = get_mol_from_smiles(self.history[state_index])
    _, mols = get_next_mols(mol, fp_length=self.fingerprint_dim)
    mol = get_fingerprint(mol, fp_length=self.fingerprint_dim)
    next_mols = np.zeros((self.max_actions, self.fingerprint_dim), dtype=np.float32)
    action_mask = np.zeros(self.max_actions, dtype=np.float32)
    if len(mols) > 0:
      next_mols[:len(mols), :] = mols
      action_mask[:len(mols)] = 1.
    return mol, next_mols, action_mask

class ReplayBuffer(object):

  def __init__(self, config: AlphaZeroConfig):
    self.config = config
    self.buffer_size = config.buffer_size
    self.batch_size = config.batch_size
    self.buffer = []

  #TODO: ALLOW UPDATING BUFFER AND SCORES WITH MULTIPLE GAMES

  def save_game(self, game):
    if len(self.buffer) >= self.buffer_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):

    # Sample games at random
    games = np.random.choice(self.buffer, size=self.batch_size)

    # For each game, select a position at random to generate targets for
    # network.  Exclude terminal states for which we don't need predictions.
    game_pos = [(g, np.random.randint(len(g.history)-1)) for g in games]

    # Get and parse inputs for each game/position
    x = [g.make_inputs(i) for (g, i) in game_pos]
    mol = [z[0] for z in x]
    next_mols = [z[1] for z in x]
    action_mask = [z[2] for z in x]

    # MCTS visit counts... shouldn't these be normalized from NN standpoint?
    pi = [g.child_visits[i] for (g, i) in game_pos]

    # Get ranked reward threshold over the entire buffer
    rewards = [g.terminal_value(-1) for g in self.buffer]
    r_alpha = np.percentile(rewards, 100.*self.config.r_alpha)

    # Compute the ranked reward for each sampled game
    v = []
    for g in games:
      value = g.terminal_value(-1)
      if value < r_alpha:
        rr = -1.
      elif value > r_alpha:
        rr = 1.
      else:
        rr = np.random.choice([-1., 1.])
      v.append(rr)

    return mol, next_mols, action_mask, v, pi, r_alpha


class Network(object):

  def __init__(self, config: AlphaZeroConfig):

    # TODO: HOW TO MAKE "UNIFORM" WHEN INITIALIZED?
    self.config = config

    kreg = tf.keras.regularizers.l2(l=1e-4)
    kini = tf.keras.initializers.Zeros()

    # Inputs
    mol = Input(shape=(self.config.fingerprint_dim,), name="mol")
    next_mols = Input(shape=(self.config.network_max_actions, self.config.fingerprint_dim,),
      name="next_mols")
    action_mask = Input(shape=(self.config.network_max_actions,), name="action_mask")

    # Shared layers
    x = Dense(self.config.network_num_units, kernel_regularizer=kreg, kernel_initializer=kini)(mol)
    x = Dense(self.config.network_num_units, kernel_regularizer=kreg, kernel_initializer=kini)(x)
    v = Dense(1, activation="tanh", name="v", kernel_regularizer=kreg, 
                kernel_initializer=kini)(x)

    # Policy head
    y = Dense(self.config.network_num_units, kernel_regularizer=kreg, kernel_initializer=kini)(mol)
    y = Dense(self.config.network_num_units, kernel_regularizer=kreg, kernel_initializer=kini)(y)
    action_embed = Dense(self.config.fingerprint_dim, kernel_regularizer=kreg, 
      kernel_initializer=kini, activation="linear", name="action_embed")(y)
    intent_vector = tf.expand_dims(action_embed, 1)
    pi_logits = tf.reduce_sum(next_mols * intent_vector, axis=2)
    inf_mask = tf.maximum(K.log(action_mask), tf.float32.min)
    pi_logits = pi_logits + inf_mask
    pi_logits = Lambda(lambda x: x, name="pi_logits")(pi_logits)

    self.model = Model(inputs=[mol, next_mols, action_mask], outputs=[v, pi_logits])

    # TODO: loss for pi should be entropy
    self.model.compile(
      optimizer="adam", 
      loss={"v": tf.keras.losses.MSE, "pi_logits": tf.nn.softmax_cross_entropy_with_logits})

    # print(self.model.summary())


  def inference(self, mol, next_mols, action_mask):
    v, pi = self.model([mol[None, :], next_mols[None, :], action_mask[None, :]])
    return v, pi


class SharedStorage(object):

  def __init__(self, config: AlphaZeroConfig):
    self._networks = {}
    self._config = config

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      return make_uniform_network(self._config)  # policy -> uniform, value -> 0.5

  def save_network(self, step: int, network: Network):
    self._networks[step] = network


##### End Helpers ########
##########################


def alphazero(config: AlphaZeroConfig):
  storage = SharedStorage(config)
  replay_buffer = ReplayBuffer(config)

  # Alternate between collecting data using the current policy, and training
  # the network with MCTS data
  log_data = defaultdict(list)
  games_played, gradient_updates = 0, 0
  for step in range(config.training_steps):
    print("alphazero step", step)
    
    # Get the latest network
    network = storage.latest_network()
  
    # Play the new games and save to buffer
    rewards = []
    for _ in tqdm(range(config.games_per_iteration)):
      game = play_game(config, network)
      replay_buffer.save_game(game)
      games_played += 1
      rewards.append(game.terminal_value(-1))
    log_data["games_played"].append(games_played)
    log_data["mean_reward"].append(np.mean(rewards))
    log_data["std_reward"].append(np.std(rewards))
    print("reward mean = {:1.5f}, std = {:1.5f}".format(
      log_data["mean_reward"][-1], log_data["std_reward"][-1]))

    # Perform backprop on batches in the replay buffer
    buffer_ratio = len(replay_buffer.buffer) / replay_buffer.buffer_size
    update_steps = int(buffer_ratio * config.gradient_steps_per_update)  
    log_data["buffer_size"].append(replay_buffer.buffer_size)
    log_data["buffer_count"].append(len(replay_buffer.buffer))
    log_data["buffer_ratio"].append(buffer_ratio)
    print("performing {} gradient steps (buffer {:1.1f}% full)".format(
      update_steps, 100.*buffer_ratio))
    
    # Get inputs
    mol, next_mols, action_mask, v, pi, r_a = replay_buffer.sample_batch()
    log_data["r_alpha"].append(r_a)
    log_data["alpha"].append(config.r_alpha)
    print("r_{}={:1.5f}".format(config.r_alpha, r_a))

    # Use the latest network
    network = storage.latest_network()

    # Perform gradient update steps on batch
    for gs in range(update_steps):
      loss = network.model.train_on_batch([mol, next_mols, action_mask], [v, pi])
      print("grad step {}, losses: {}".format(gs, loss))
      gradient_updates += 1
    log_data["gradient_updates"].append(gradient_updates)
    log_data["loss_total"].append(loss[0])
    log_data["mse"].append(loss[1])
    log_data["cross_entropy"].append(loss[2])
    print("iteration losses (tot,mse,entropy)", loss)

    # Play one evaluation game with exploration off, don't save to replay buffer
    game = play_game(config, network, explore=False)
    log_data["eval_mol"].append(game.history[-1])
    log_data["eval_reward"].append(game.terminal_value(-1))
    print("policy eval", log_data["eval_mol"][-1], log_data["eval_reward"][-1])

    # Save network and log dataframe
    storage.save_network(0, network)
    pd.DataFrame(log_data).to_csv("log.csv")
    
  return storage.latest_network()


##################################
####### Part 1: Self-Play ########

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network, explore=True):
  game = Game(config)
  game_step = 0
  while not game.terminal() and len(game.history) < config.max_atoms:
    action, root = run_mcts(config, game, network, explore)
    game.apply(action)
    game.store_search_statistics(root)
    game_step += 1
  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network, explore=True):
  
  root = Node(0, game.history[0])
  evaluate(root, game, network)
  # if explore:
  #   add_exploration_noise(config, root)

  for _ in range(config.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    mcts_step = 0
    while node.expanded():
      action, node = select_child(config, node)
      scratch_game.apply(action)
      search_path.append(node)
      mcts_step += 1

    if scratch_game.terminal():
      value = scratch_game.terminal_value(-1)
    else:
      value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value)
  
  return select_action(config, game, root, explore), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node, explore=True):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.items()]
  # In original pseudocode, you only did this for short games, then switched to
  # the action with the most visits after num_sampling_moves moves.
  if explore:
    _, action = softmax_sample(visit_counts)
  else:
    action = np.argmax([x[0] for x in visit_counts])
  return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
  _, action, child = max((ucb_score(config, node, child), action, child)
                         for action, child in node.children.items())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
  pb_c = np.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
  mol, next_mols, action_mask = game.make_inputs(-1)
  value, policy_logits = network.inference(mol, next_mols, action_mask)
  value = float(tf.squeeze(value))
  policy_logits = tf.squeeze(policy_logits)

  # Expand the node.
  policy = {a: np.exp(float(policy_logits[a])) for a in game.legal_actions()}
  policy_sum = float(sum(policy.values()))
  for ix, (action, p) in enumerate(policy.items()):
    node.children[action] = Node(p/policy_sum, game.next_mols[ix])
  return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float):
  for node in search_path:
    node.value_sum += value
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = node.children.keys()
  noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

# # Stubs to make the typechecker happy, should not be included in pseudocode
# # for the paper.
def softmax_sample(d):
  d = np.exp([x[0] for x in d])
  d_sum = d.sum()
  return None, np.random.choice(range(len(d)), size=1, p=d/d_sum)[0]


def launch_job(f, *args):
  f(*args)


def make_uniform_network(config: AlphaZeroConfig):
  return Network(config)


if __name__ == "__main__":

  config = AlphaZeroConfig()
  print(config.__dict__)
  alphazero(config)

  #network = Network(config)
  #play_game(config, network)

  #game = Game(config)
  #run_mcts(config, game, network)
