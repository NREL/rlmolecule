import tensorflow as tf
from tensorflow.keras import layers


def policy_model(
    input_dim: int = 256,
    num_layers: int = 3,
    layer_dim: int = 256
) -> tf.keras.Model:
    """ Constructs a policy model that predicts value, pi_logits from a batch of molecule inputs. Main model used in
    policy training and loading weights"""

    # Define inputs
    fp_input = layers.Input(shape=[input_dim,], dtype=tf.float32, name='fingerprint')  # batch_size, embed_dim

    x = layers.Dense(layer_dim, activation="relu")(fp_input)
    for _ in range(num_layers - 1):
        x = layers.Dense(layer_dim, activation="relu")(x)

    value_logit = layers.Dense(1)(x)
    pi_logit = layers.Dense(1)(x)

    return tf.keras.Model([fp_input], [value_logit, pi_logit], name='policy_model')
