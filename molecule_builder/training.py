import glob
import os
import pickle
import time
import numpy as np
import tensorflow as tf

from config import AlphaZeroConfig
from network import Network

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
    for iteration in range(CONFIG.training_iterations):
        mol, next_mols, action_mask, v, pi = sample_batch(buffer_dir)
        #for gs in range(CONFIG.gradient_steps_per_batch):
            #loss = network.model.train_on_batch([mol, next_mols, action_mask], [v, pi])
        checkpoint_filepath = os.path.join(model_dir,'cp.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                            save_weights_only=True,
                                                            save_best_only=False,
                                                            verbose=0,
                                                            period=2)
        history = network.model.fit([np.asarray(mol), np.asarray(next_mols), np.asarray(action_mask)], 
                                    [np.asarray(v), np.asarray(pi)],
                                    epochs=CONFIG.gradient_steps_per_batch,
                                    callbacks=[cp_callback],
                                    verbose=0)
        network.model.save_weights(checkpoint_filepath.format(epoch=0))
        #network.model.save(os.path.join(model_dir,'model_{}.h5'.format(time.strftime("%Y%m%d-%H%M%S"))))

if __name__ == "__main__":

    current_path = os.getcwd()
    buffer_dir = os.path.join(current_path, 'pickled_objects')
    model_dir = os.path.join(current_path, 'saved_models')

    network = Network(model_dir)
    network.compile()
    train_model(network, buffer_dir, model_dir)