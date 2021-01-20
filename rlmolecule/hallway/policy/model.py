from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers


def policy_model(hallway_size: int,
                 max_steps: int,
                 features: int = 8,
                 hidden_layers: int = 3,
                 hidden_dim: int = 16) -> tf.keras.Model:

    # Position input and embedding
    position = layers.Input([None, 1], dtype=tf.int64, name="position")
    xp = layers.Embedding(hallway_size, features, name="position_emb")(position)
    xp = tf.squeeze(xp, axis=2)  # so we can concatenate with steps later

    # Steps input and normalization by max_steps (to [0, 1])
    steps = layers.Input([None, 1], dtype=tf.float32, name="steps")
    xs = layers.Lambda(lambda x: x / float(max_steps), name="steps_norm")(steps)
    
    # Dense layers and concatenation
    for layer in range(hidden_layers):
        xp = layers.Dense(hidden_dim, activation="relu")(xp)
        xs = layers.Dense(hidden_dim, activation="relu")(xs)
    x = layers.Concatenate()([xp, xs])

    value_logit = layers.Dense(1)(x)
    pi_logit = layers.Dense(1)(x)

    return tf.keras.Model([position, steps], [value_logit, pi_logit], name="policy_model")


class PolicyWrapper(layers.Layer):

    def __init__(self,
                 hallway_size: int,
                 max_steps: int,
                 features: int = 8,
                 hidden_layers: int = 3,
                 hidden_dim: int = 16,
                 **kwargs):
        super().__init__(**kwargs)
        self.hallway_size = hallway_size
        self.max_steps = max_steps
        self.features = features
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.policy_model = policy_model(self.hallway_size,
                                         self.max_steps,
                                         self.features,
                                         self.hidden_layers,
                                         self.hidden_dim)

    def call(self, inputs, mask=None):

        position, steps = inputs

        # Get the batch and action dimensions
        shape = tf.shape(position)
        batch_size = shape[0]
        max_actions_per_node = shape[1]

        # Flatten the inputs for running individually through the policy model
        position_flat = tf.reshape(position, [batch_size * max_actions_per_node, -1])
        steps_flat = tf.reshape(steps, [batch_size * max_actions_per_node, -1])

        # Get the flat value and prior_logit predictions
        flat_values_logits, flat_prior_logits = self.policy_model([position_flat, steps_flat])

        # We put the parent node first in our batch inputs, so this slices
        # the value prediction for the parent
        value_preds = tf.reshape(flat_values_logits, [batch_size, max_actions_per_node, -1])[:, 0, 0]

        # Next we get a mask to see where we have valid actions and replace priors for
        # invalid actions with negative infinity (these get zeroed out after softmax).
        # We also only return prior_logits for the child nodes (not the first entry)
        action_mask = tf.reduce_any(tf.not_equal(position, -1), axis=-1)  # -1 is the padding element
        prior_logits = tf.reshape(flat_prior_logits, [batch_size, max_actions_per_node])
        masked_prior_logits = tf.where(action_mask, prior_logits,
                                       tf.ones_like(prior_logits) * prior_logits.dtype.min)[:, 1:]

        return value_preds, masked_prior_logits

    
def build_policy_trainer(hallway_size: int,
                         max_steps: int,
                         features: int = 8,
                         hidden_layers: int = 3,
                         hidden_dim: int = 16) -> tf.keras.Model:
    """Builds a keras model that expects [bsz, actions] molecules as inputs and predicts batches of value scores and
    prior logits

    :return: the built keras model
    """
    position = layers.Input(shape=[None, 1], dtype=tf.int64, name="position")
    steps = layers.Input(shape=[None, 1], dtype=tf.int64, name="steps")

    value_preds, masked_prior_logits = PolicyWrapper(
        hallway_size, max_steps, features, hidden_layers, hidden_dim)([position, steps])

    policy_trainer = tf.keras.Model([position, steps], [value_preds, masked_prior_logits])
    
    return policy_trainer


def build_policy_evaluator(checkpoint_filepath: Optional[str] = None) -> Tuple[tf.function, Optional[str]]:
    """Builds (or loads from a checkpoint) a model that expects a single batch of input molecules.

    :param checkpoint_filepath: A filename specifying a checkpoint from a saved policy iteration
    :return: The policy_model layer of the loaded or initalized molecule.
    """
    policy_trainer = build_policy_trainer()

    latest = tf.train.latest_checkpoint(checkpoint_filepath) if checkpoint_filepath else None
    if latest:
        policy_trainer.load_weights(latest)

    policy_model_layer = policy_trainer.layers[-1].policy_model
    policy_predictor = tf.function(experimental_relax_shapes=True)(policy_model_layer.predict_step)

    return policy_predictor, latest

