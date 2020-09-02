import glob
import os
import pickle
import time
import numpy as np

from config import AlphaZeroConfig

CONFIG = AlphaZeroConfig()

def sample_batch(buffer_dir):

    # Get last N pickled elements and append them to a list
    glob_pattern = os.path.join(buffer_dir, '*[0-9].pickle')
    file_list = sorted(glob.glob(glob_pattern, recursive=False))
    num_samples = min(len(file_list), CONFIG.buffer_max_size)
    game_list = []

    for name in file_list[-num_samples:]:
        with open(name, 'rb') as f:
            game_list.append(pickle.load(f))
    games = np.random.choice(game_list, size=CONFIG.batch_size)
    game_pos = [(g, np.random.randint(len(g["mol_smiles"])-1)) for g in games]

    mol = [z["network_inputs"]["mol"][i] for (z, i) in game_pos]
    next_mols = [z["network_inputs"]["next_mols"][i] for (z, i) in game_pos]
    action_mask = [z["network_inputs"]["action_mask"][i] for (z, i) in game_pos]

    pi = [z["network_inputs"]["pi"][i] for (z, i) in game_pos]

    # Get ranked reward threshold over the entire buffer
    rewards = [z[0]["reward"] for z in game_pos]
    r_alpha = np.percentile(rewards, 100.*CONFIG.ranked_reward_alpha)

    # Compute the ranked reward for each sampled game
    v = []
    for (z, _) in game_pos:
        value = z["reward"]
        if value < r_alpha:
            rr = -1.
        elif value > r_alpha:
            rr = 1.
        else:
            rr = np.random.choice([-1., 1.])
        v.append(rr)
    
    return mol, next_mols, action_mask, v, pi

def train_model(network, buffer_dir, model_dir):
    for _ in range(CONFIG.training_iterations):
        mol, next_mols, action_mask, v, pi = sample_batch(buffer_dir)
        for gs in range(CONFIG.gradient_steps_per_batch):
            loss = network.model.train_on_batch([mol, next_mols, action_mask], [v, pi])
            print("grad step:{}, loss:{}".format(gs, loss))
        network.model.save(os.path.join(model_dir,'model_{}.h5'.format(time.strftime("%Y%m%d-%H%M%S"))))

if __name__ == "__main__":
    #mol, next_mols, action_mask, v, pi = sample()
    current_path = os.getcwd()
    buffer_dir = os.path.join(current_path, 'pickled_objects')
    glob_pattern = os.path.join(buffer_dir, '*[0-9].pickle')
    file_list = sorted(glob.glob(glob_pattern, recursive=False))
    print(file_list)

    num_samples = 2

    for name in file_list[-num_samples:]: 
        with open(name, 'rb') as f:
            new_data = pickle.load(f)
        print(name) 
