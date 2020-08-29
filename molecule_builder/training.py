import pickle
import numpy as np
import random
import os
import glob

from config import AlphaZeroConfig
from network import Network

CONFIG = AlphaZeroConfig()
""""""

def sample():

    # Get last N pickled elements and append them to a list
    sample = CONFIG.buffer_max_size
    file_list = sorted(glob.glob('*[0-9].pickle'))
    game_list = []

    for name in file_list[-sample:]:
        with open(name, 'rb') as f:
            new_data = pickle.load(f)
            game_list.append(new_data)
    move_sum = float(sum(len(g["mol_smiles"]) for g in game_list))
    games = np.random.choice(
                game_list,
                size=CONFIG.batch_size,
                p=[len(g["mol_smiles"]) / move_sum for g in game_list]
            )
    game_pos = [(g, np.random.randint(len(g["mol_smiles"])-1)) for g in games]

    mol = [z["network_inputs"]["mol"][i] for (z, i) in game_pos]
    next_mols = [z["network_inputs"]["next_mols"][i] for (z, i) in game_pos]
    action_mask = [z["network_inputs"]["action_mask"][i] for (z, i) in game_pos]

    pi = [z["network_inputs"]["pi"][i] for (z, i) in game_pos]

    # Get ranked reward threshold over the entire buffer
    rewards = [z["reward"] for (z, i) in game_pos]
    r_alpha = np.percentile(rewards, 100.*CONFIG.ranked_reward_alpha)

    # Compute the ranked reward for each sampled game
    v = []
    for (z, i) in game_pos:
      value = z["reward"]
      if value < r_alpha:
        rr = -1.
      elif value > r_alpha:
        rr = 1.
      else:
        rr = np.random.choice([-1., 1.])
      v.append(rr)
    
    return mol, next_mols, action_mask, v, pi

def model_training(network, args):
    for iteration in range(CONFIG.training_iterations):
        mol, next_mols, action_mask, v, pi = sample(args)
        for _ in range(CONFIG.gradient_steps_per_batch):
            result = network.model.train_on_batch([mol, next_mols, action_mask], [v, pi])
        network.save()

if __name__ == "__main__":
    #mol, next_mols, action_mask, v, pi = sample()
    sample = 2
    file_list = sorted(glob.glob('*[0-9].pickle'))

    for name in file_list[-sample:]: 
        with open(name, 'rb') as f:
            new_data = pickle.load(f)
        print(name) 
