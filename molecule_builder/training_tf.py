import glob
import os
import pickle
import random
import time
import numpy as np
import tensorflow as tf

from config import AlphaZeroConfig
from network import Network

CONFIG = AlphaZeroConfig()

def get_r_alpha():
    current_path = os.getcwd()
    buffer_dir = os.path.join(current_path, 'pickled_objects')
    glob_pattern = os.path.join(buffer_dir, '*[0-9].pickle')
    files = glob.glob(glob_pattern)
    most_recent = sorted(files, key=os.path.getctime, reverse=True)[:CONFIG.buffer_max_size]
    reward_list = []

    for name in most_recent:
        with open(name, 'rb') as f:
            data = pickle.load(f)
            reward_list.append(data["reward"])
    
    return np.percentile(reward_list, 100.*CONFIG.ranked_reward_alpha)

def get_ranked_reward(rew):

    r_alpha = get_r_alpha()
    
    if rew < r_alpha:
        v = -1.
    elif rew > r_alpha:
        v = 1.
    else:
        v = np.random.choice([-1., 1.])
    
    return np.array(v, dtype=np.float32)

def parser_fn(filename):
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        game_pos = np.random.randint(len(data['mol_smiles']) - 1)
                
        inputs = {
            "mol": data['network_inputs']['mol'][game_pos],
            "next_mols": data['network_inputs']['next_mols'][game_pos],
            "action_mask": data['network_inputs']['action_mask'][game_pos]
            }
        outputs = {
            "v": get_ranked_reward(data['reward']),
            "pi_logits": data['network_inputs']['pi'][game_pos]
            }

        return (inputs, outputs)

def sample_batch(buffer_dir):

    glob_pattern = os.path.join(buffer_dir, '*[0-9].pickle')

    def file_generator():
        files = glob.glob(glob_pattern)
        most_recent = sorted(files, key=os.path.getctime, reverse=True)[:CONFIG.buffer_max_size]
        for filename in most_recent:
            yield parser_fn(filename)

    file_list = tf.data.Dataset.from_generator(
                    file_generator, 
                    output_types=({
                        "mol":  tf.float32,
                        "next_mols": tf.float32,
                        "action_mask": tf.float32
                    },{
                        "pi_logits":  tf.float32,
                        "v": tf.float32,
                    }),
                    output_shapes=({
                        "mol": [None, CONFIG.fingerprint_dim],
                        "next_mols": [None, CONFIG.max_next_mols, CONFIG.fingerprint_dim],
                        "action_mask": [None, CONFIG.max_next_mols]
                    },{
                        "pi_logits": [None, CONFIG.max_next_mols],
                        "v": [],
                    })
                    ).repeat().shuffle(buffer_size=CONFIG.batch_size, reshuffle_each_iteration=True).\
                        padded_batch(
                        batch_size=CONFIG.batch_size,
                            padded_shapes=({
                                "mol":  [-1, CONFIG.fingerprint_dim],
                                "next_mols": [-1, CONFIG.max_next_mols, CONFIG.fingerprint_dim],
                                "action_mask": [-1, CONFIG.max_next_mols]
                            },{
                                "pi_logits": [-1, CONFIG.max_next_mols],
                                "v": [],
                            }),
                            padding_values=({
                                "mol":  0.,
                                "next_mols": 0.,
                                "action_mask": 0.
                            },{
                                "pi_logits": 0.,
                                "v": 0.,
                            }),
                            drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    
    return file_list

def train_model(network, buffer_dir, model_dir):
    dataset = sample_batch(buffer_dir)
    full_batches = CONFIG.buffer_max_size // CONFIG.batch_size
    checkpoint_filepath = os.path.join(model_dir,'cp.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                        save_weights_only=True,
                                                        save_best_only=False,
                                                        verbose=0
                                                    )
    history = network.model.fit(dataset,
                                steps_per_epoch=full_batches,
                                epochs=CONFIG.training_iterations,
                                callbacks=[cp_callback],
                                verbose=True)

if __name__ == "__main__":

    current_path = os.getcwd()
    buffer_dir = os.path.join(current_path, 'pickled_objects')
    model_dir = os.path.join(current_path, 'saved_models')

    network = Network(model_dir)
    network.compile()
    train_model(network, buffer_dir, model_dir)