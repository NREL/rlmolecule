from typing import Optional, Tuple

import nfp
import tensorflow as tf
from tensorflow.keras import layers

from rlmolecule.molecule.policy.preprocessor import MolPreprocessor, load_preprocessor


def policy_model(preprocessor: Optional[MolPreprocessor] = None,
                 features: int = 64,
                 num_heads: int = 4,
                 num_messages: int = 3) -> tf.keras.Model:
    """ Constructs a policy model that predicts value, pi_logits from a batch of molecule inputs. Main model used in
    policy training and loading weights

    :param preprocessor: a MolPreprocessor class for initializing the embedding matrices
    :param features: Size of network hidden layers
    :param num_heads: Number of global state attention heads. Must be a factor of `features`
    :param num_messages: Number of message passing layers
    :return: The constructed policy model
    """
    if preprocessor is None:
        preprocessor = load_preprocessor()

    # Define inputs
    atom_class = layers.Input(shape=[None], dtype=tf.int64, name='atom')  # batch_size, num_atoms
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')  # batch_size, num_bonds
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')  # batch_size, num_bonds, 2

    input_tensors = [atom_class, bond_class, connectivity]

    # Initialize the atom states
    atom_state = layers.Embedding(preprocessor.atom_classes, features,
                                  name='atom_embedding', mask_zero=True)(atom_class)

    # Initialize the bond states
    bond_state = layers.Embedding(preprocessor.bond_classes, features,
                                  name='bond_embedding', mask_zero=True)(bond_class)

    units = features // num_heads
    global_state = nfp.GlobalUpdate(units=units, num_heads=num_heads)(
        [atom_state, bond_state, connectivity])

    for _ in range(num_messages):  # Do the message passing
        new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
        atom_state = layers.Add()([atom_state, new_atom_state])

        new_global_state = nfp.GlobalUpdate(units=units, num_heads=num_heads)(
            [atom_state, bond_state, connectivity, global_state])
        global_state = layers.Add()([global_state, new_global_state])

    value_logit = layers.Dense(1)(global_state)
    pi_logit = layers.Dense(1)(global_state)

    return tf.keras.Model(input_tensors, [value_logit, pi_logit], name='policy_model')
#
#
# class PolicyWrapper(layers.Layer):
#
#     def __init__(self,
#                  preprocessor: Optional[MolPreprocessor] = None,
#                  features: int = 64,
#                  num_heads: int = 4,
#                  num_messages: int = 3,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.preprocessor = preprocessor
#         self.features = features
#         self.num_heads = num_heads
#         self.num_messages = num_messages
#
#     def build(self, input_shape):
#         self.policy_model = policy_model(self.preprocessor,
#                                          self.features,
#                                          self.num_heads,
#                                          self.num_messages)
#
#     def call(self, inputs, mask=None):
#         atom, bond, connectivity = inputs
#
#         # Get the batch and action dimensions
#         atom_shape = tf.shape(atom)
#         batch_size = atom_shape[0]
#         max_actions_per_node = atom_shape[1]
#
#         # Flatten the inputs for running individually through the policy model
#         atom_flat = tf.reshape(atom, [batch_size * max_actions_per_node, -1])
#         bond_flat = tf.reshape(bond, [batch_size * max_actions_per_node, -1])
#         connectivity_flat = tf.reshape(connectivity, [batch_size * max_actions_per_node, -1, 2])
#
#         # Get the flat value and prior_logit predictions
#         flat_values_logits, flat_prior_logits = self.policy_model([atom_flat, bond_flat, connectivity_flat])
#
#         # We put the parent node first in our batch inputs, so this slices
#         # the value prediction for the parent
#         value_preds = tf.reshape(flat_values_logits, [batch_size, max_actions_per_node, -1])[:, 0, 0]
#
#         # Next we get a mask to see where we have valid actions and replace priors for
#         # invalid actions with negative infinity (these get zeroed out after softmax).
#         # We also only return prior_logits for the child nodes (not the first entry)
#         action_mask = tf.reduce_any(tf.not_equal(atom, 0), axis=-1)  # zero is the padding element
#         prior_logits = tf.reshape(flat_prior_logits, [batch_size, max_actions_per_node])
#         masked_prior_logits = tf.where(action_mask, prior_logits,
#                                        tf.ones_like(prior_logits) * prior_logits.dtype.min)[:, 1:]
#
#         return value_preds, masked_prior_logits
#
#
# def build_policy_trainer(preprocessor: Optional[MolPreprocessor] = None,
#                          features: int = 64,
#                          num_heads: int = 4,
#                          num_messages: int = 3) -> tf.keras.Model:
#     """Builds a keras model that expects [bsz, actions] molecules as inputs and predicts batches of value scores and
#     prior logits
#
#     :return: the built keras model
#     """
#     atom_class = layers.Input(shape=[None, None], dtype=tf.int64, name='atom')
#     bond_class = layers.Input(shape=[None, None], dtype=tf.int64, name='bond')
#     connectivity = layers.Input(shape=[None, None, 2], dtype=tf.int64, name='connectivity')
#
#     value_preds, masked_prior_logits = PolicyWrapper(
#         preprocessor, features, num_heads, num_messages)([atom_class, bond_class, connectivity])
#
#     policy_trainer = tf.keras.Model([atom_class, bond_class, connectivity], [value_preds, masked_prior_logits])
#     # policy_trainer.compile(
#     #     optimizer=tf.keras.optimizers.Adam(config.policy_lr),  # Do AZ list their optimizer?
#     #     loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True), KLWithLogits()])
#
#     return policy_trainer
#
#
# def build_policy_evaluator(checkpoint_filepath: Optional[str] = None) -> Tuple[tf.function, Optional[str]]:
#     """Builds (or loads from a checkpoint) a model that expects a single batch of input molecules.
#
#     :param checkpoint_filepath: A filename specifying a checkpoint from a saved policy iteration
#     :return: The policy_model layer of the loaded or initalized molecule.
#     """
#     policy_trainer = build_policy_trainer()
#
#     latest = tf.train.latest_checkpoint(checkpoint_filepath) if checkpoint_filepath else None
#     if latest:
#         policy_trainer.load_weights(latest)
#
#     policy_model_layer = policy_trainer.layers[-1].policy_model
#     policy_predictor = tf.function(experimental_relax_shapes=True)(policy_model_layer.predict_step)
#
#     return policy_predictor, latest
