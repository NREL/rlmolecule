import glob
import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution

from config import AlphaZeroConfig
from network import Network

CONFIG = AlphaZeroConfig()

def parser_fn(filename):
    with open(filename.numpy().decode("utf-8"), 'rb') as f:
        return pickle.load(f)

def py_func(x):
    d = parser_fn(x)
    return [
        d.get("network_inputs").get("mol"),
        d.get("network_inputs").get("next_mols"),
        d.get("network_inputs").get("action_mask"),
        d.get("network_inputs").get("pi"),
        d.get("mol_smiles"),
        d.get("reward")
    ]

def ds_map_fn(x):
    flattened_output = tf.py_function(py_func, [x], [tf.float32, tf.float32, tf.float32, tf.float32, tf.string, tf.float32])
    return [{
        "network_inputs": {
            "mol":  flattened_output[0],
            "next_mols": flattened_output[1],
            "action_mask": flattened_output[2],
            "pi":  flattened_output[3],
        },
        "mol_smiles": flattened_output[4],
        "reward": flattened_output[5],
    }]

def sample_batch(buffer_dir):

    # Use tf.data.Dataset to retrieve pickled objects
    glob_pattern = os.path.join(buffer_dir, '*[0-9].pickle')
    file_list = tf.data.Dataset.list_files(glob_pattern, seed=1234, shuffle=False)
    file_list_len = tf.data.experimental.cardinality(file_list).numpy()
    
    
    # Set number of sampled games
    num_samples = min(file_list_len, CONFIG.buffer_max_size)
    
    # Create a list of the sampled games, unpickled (?)
    def get_last_n(x, y):
        return x >= file_list_len - num_samples
    
    recover = lambda x, y: y

    file_list = file_list.enumerate()\
        .filter(get_last_n)\
            .map(recover)
    
    file_list = file_list.map(ds_map_fn)

    # TODO: The following line is a bottleneck (around 8 seconds), 
    # so far haven't found a way to avoid transforming MapDataset to list
    start_time = time.time()
    game_list = list(file_list)
    end_time = time.time()
    print("Time taken for this loop: ",end_time - start_time)
    
    games = np.random.choice(len(game_list), size=CONFIG.batch_size)
    game_pos = [(g, np.random.randint(len(game_list[g][0]["mol_smiles"].numpy())-1)) for g in games]

    mol = [game_list[z][0]["network_inputs"]["mol"][i].numpy() for (z, i) in game_pos]
    next_mols = [game_list[z][0]["network_inputs"]["next_mols"][i].numpy() for (z, i) in game_pos]
    action_mask = [game_list[z][0]["network_inputs"]["action_mask"][i].numpy() for (z, i) in game_pos]

    pi = [game_list[z][0]["network_inputs"]["pi"][i].numpy() for (z, i) in game_pos]

    # Get ranked reward threshold over the entire buffer
    rewards = [game_list[z[0]][0]["reward"].numpy() for z in game_pos]
    r_alpha = np.percentile(rewards, 100.*CONFIG.ranked_reward_alpha)

    # Compute the ranked reward for each sampled game
    v = []
    for z in game_pos:
        value = game_list[z[0]][0]["reward"].numpy()
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
                                                            verbose=0
                                                        )
        history = network.model.fit([np.asarray(mol), np.asarray(next_mols), np.asarray(action_mask)], 
                                    [np.asarray(v), np.asarray(pi)],
                                    epochs=CONFIG.gradient_steps_per_batch,
                                    callbacks=[cp_callback],
                                    verbose=1)
        network.model.save_weights(checkpoint_filepath)
        #network.model.save(os.path.join(model_dir,'model_{}.h5'.format(time.strftime("%Y%m%d-%H%M%S"))))

if __name__ == "__main__":

    current_path = os.getcwd()
    buffer_dir = os.path.join(current_path, 'pickled_objects')
    model_dir = os.path.join(current_path, 'saved_models')

    network = Network(model_dir)
    network.compile()
    train_model(network, buffer_dir, model_dir)