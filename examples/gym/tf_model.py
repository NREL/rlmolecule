from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def policy_model(obs_dim: int,
                 hidden_layers: int = 1,
                 hidden_dim: int = 16,
                 activation: str = "relu",
                 input_dtype: str = "float64") -> tf.keras.Model:

    # Position input
    input_dtype = tf.dtypes.as_dtype(input_dtype)
    obs = layers.Input(shape=[obs_dim, ], dtype=input_dtype, name="obs")

    # Dense layers and concatenation
    x = layers.Dense(hidden_dim, activation=activation)(obs)
    for _ in range(hidden_layers-1):
        x = layers.Dense(hidden_dim, activation=activation)(x)
    x = layers.BatchNormalization()(x)

    value_logit = layers.Dense(1, name="value")(x)
    pi_logit = layers.Dense(1, name="prior")(x)

    return tf.keras.Model([obs], [value_logit, pi_logit], name="policy_model")

def policy_model_cnn(obs_type: str = "RGB",
                     obs_dim: tuple = (84, 84, 4),
                     hidden_layers: int = 1,
                     conv_layers: int = 2,
                     filters_dim: list = [32, 64],
                     kernel_dim: list = [8, 4],
                     strides_dim: list = [4, 3],
                     hidden_dim: int = 1024,
                     activation: str = "relu") -> tf.keras.Model:

    obs = layers.Input(shape=obs_dim, dtype=tf.float32, name="obs")
    if obs_type == "RGB":
        x = layers.Lambda(lambda layer: layer / 255)(obs) # normalize by 255, if necessary

    # Convolutions on the frames on the screen
    x = layers.Conv2D(filters_dim[0], (kernel_dim[0],kernel_dim[0]), strides_dim[0], activation=activation)(x)
    for i in range(conv_layers-1):
        x = layers.Conv2D(filters_dim[i], (kernel_dim[i],kernel_dim[i]), strides_dim[i], activation=activation)(x)
    x = layers.Flatten()(x)

    for _ in range(hidden_layers):
        x = layers.Dense(hidden_dim, activation=activation)(x)
    
    value_logit = layers.Dense(1, name="value")(x)
    pi_logit = layers.Dense(1, name="prior")(x)

    return tf.keras.Model([obs], [value_logit, pi_logit], name="policy_model_cnn")



def policy_model_2(filters,
                   kernel_size,
                   strides,
                   obs_dim: Tuple[int],
                   hidden_layers: int = 2,
                   hidden_dim: int = 128,
                   activation: str = "relu") -> tf.keras.Model:

    obs = layers.Input(shape=obs_dim, dtype=tf.float64, name="obs")
    #steps = layers.Input(shape=(1,), dtype=tf.float64, name="steps")
    
    # Convolutions on the frames on the screen
    x = layers.Conv2D(
        filters[0], 
        (kernel_size[0],kernel_size[0]),
        strides[0],
        activation=activation)(obs)

    for i in range(1, len(filters)):
        x = layers.Conv2D(
                filters[i],
                (kernel_size[i],kernel_size[i]),
                strides[i],
                activation=activation)(x)

    x = layers.Flatten()(x)

    #x = layers.Concatenate()((x, steps))

    for _ in range(hidden_layers):
        x = layers.Dense(hidden_dim, activation=activation)(x)
        x = layers.BatchNormalization()(x)
    
    value_logit = layers.Dense(1, name="value")(x)
    pi_logit = layers.Dense(1, name="prior")(x)

    #return tf.keras.Model([obs, steps], [value_logit, pi_logit], name="policy_model")
    return tf.keras.Model([obs], [value_logit, pi_logit], name="policy_model")



def scalar_obs_policy(
        hidden_layers: int = 2,
        hidden_dim: int = 256,
        activation: str = "relu") -> tf.keras.Model:

    obs = layers.Input(shape=[1, 2], dtype=tf.float64, name="obs")

    x = layers.Dense(hidden_dim, activation=activation)(obs)
    for _ in range(hidden_layers):
        x = layers.Dense(hidden_dim, activation=activation)(x)
    
    x = layers.BatchNormalization()(x)
    
    value_logit = layers.Dense(1, name="value")(x)
    pi_logit = layers.Dense(1, name="prior")(x)

    return tf.keras.Model([obs], [value_logit, pi_logit], name="policy_model")


