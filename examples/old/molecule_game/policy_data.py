import io

import nfp
import numpy as np
import tensorflow as tf

import molecule_game.config as config


def parse_binary_data(binary_data, reward):
    """ Use io and numpy to parse the binary data from postgresQL
    """
    with io.BytesIO(binary_data.numpy()) as f:
        parsed_data = dict(np.load(f, allow_pickle=True).items())
    
    # This is something we could talk about; but I'm wondering if the best
    # loss function for a boolean reward is a binary crossentropy
    if reward == -1:
        reward = 0
    
    visit_probs = parsed_data.pop('visit_probs')
    return (parsed_data['atom'], parsed_data['bond'],
            parsed_data['connectivity'], int(reward), visit_probs)


def parse_data_tf(binary_data, reward):
    """tf.py_func wants a flat list of outputs, but here we restructure to
    keras's desired (inputs, outputs) format"""
    atom, bond, connectivity, reward, visit_probs = tf.py_function(
        parse_binary_data, inp=[binary_data, reward],
        Tout=[tf.int64, tf.int64, tf.int64, tf.int64, tf.float32])
    
    # The py_func doesn't provide tensor shapes, and we'll need these for the
    # padded batch operation
    atom.set_shape([None, None])
    bond.set_shape([None, None])
    connectivity.set_shape([None, None, 2])
    reward.set_shape([])
    visit_probs.set_shape([None])
    
    return ({'atom': atom, 'bond': bond, 'connectivity': connectivity},
            (reward, visit_probs))


def create_dataset(sql_generator):
    """ Given a generator that yields data (bytes), ranked_rewards (float) pairs,
    zip these together into a tensorflow dataset.
    """
    
    dataset = tf.data.Dataset.from_generator(sql_generator, output_types=(tf.string, tf.float32)) \
        .repeat() \
        .shuffle(config.policy_buffer_max_size) \
        .map(parse_data_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .padded_batch(config.batch_size,
                      padding_values=({'atom': nfp.zero, 'bond': nfp.zero, 'connectivity': nfp.zero}, (nfp.zero, 0.))) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset
