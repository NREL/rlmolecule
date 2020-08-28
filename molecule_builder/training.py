import pickle
import numpy as np
import random

from config import AlphaZeroConfig
from network import Network

CONFIG = AlphaZeroConfig()
""""""
def sample():

    # Get last N pickled elements and append them to a list
    sample = np.arange(CONFIG.num_rollouts)[-CONFIG.batch_size:]
    game_list = []

    for i in sample:
        with open('game_{}.pickle'.format(i), 'rb') as f:
            new_data = pickle.load(f)
            game_list.append(new_data)
    # compute ranked reward here --> compute r_alpha
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

    v = [z["reward"] for (z, i) in game_pos] # here put z (ranked reward)
    pi = [z["network_inputs"]["pi"][i] for (z, i) in game_pos]
    
    return mol, next_mols, action_mask, v, pi

def model_training(network):
    mol, next_mols, action_mask, v, pi = sample()
    network.compile()
    result = network.model.train_on_batch([mol, next_mols, action_mask], [v, pi])

if __name__ == "__main__":
    mol, next_mols, action_mask, v, pi = sample()
