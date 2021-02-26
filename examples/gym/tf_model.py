import tensorflow as tf
from tensorflow.keras import layers


def policy_model(obs_dim: int,
                 hidden_layers: int = 1,
                 hidden_dim: int = 16,
                 activation: str = "relu") -> tf.keras.Model:

    # Position input
    obs = layers.Input(shape=[obs_dim, ], dtype=tf.float32, name="obs")

    # Dense layers and concatenation
    x = layers.Dense(hidden_dim, activation=activation)(obs)
    for _ in range(hidden_layers-1):
        x = layers.Dense(hidden_dim, activation=activation)(x)
    x = layers.BatchNormalization()(x)

    value_logit = layers.Dense(1, name="value")(x)
    pi_logit = layers.Dense(1, name="prior")(x)

    return tf.keras.Model([obs], [value_logit, pi_logit], name="policy_model")

