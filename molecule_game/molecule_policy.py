import nfp
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.losses import (
    LossFunctionWrapper,
    losses_utils,
    )

import molecule_game.config as config
from molecule_game.mol_preprocessor import preprocessor


# two models:
# first, a policy model that predicts value, pi_logits from a batch of molecule inputs
# Then, a wrapper model that expects batches of games and normalizes logit values

def policy_model():
    # Define inputs
    atom_class = layers.Input(shape=[None], dtype=tf.int64, name='atom')  # batch_size, num_atoms
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')  # batch_size, num_bonds
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')  # batch_size, num_bonds, 2
    
    input_tensors = [atom_class, bond_class, connectivity]
    
    # Initialize the atom states
    atom_state = layers.Embedding(preprocessor.atom_classes, config.features,
                                  name='atom_embedding', mask_zero=True)(atom_class)
    
    # Initialize the bond states
    bond_state = layers.Embedding(preprocessor.bond_classes, config.features,
                                  name='bond_embedding', mask_zero=True)(bond_class)
    
    units = config.features // config.num_heads
    global_state = nfp.GlobalUpdate(units=units, num_heads=config.num_heads)(
        [atom_state, bond_state, connectivity])
    
    for _ in range(config.num_messages):  # Do the message passing
        bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
        atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
        global_state = nfp.GlobalUpdate(units=units, num_heads=config.num_heads)(
            [atom_state, bond_state, connectivity, global_state])
    
    value_logit = layers.Dense(1)(global_state)
    pi_logit = layers.Dense(1)(global_state)
    
    return tf.keras.Model(input_tensors, [value_logit, pi_logit], name='policy_model')


def kl_with_logits(y_true, y_pred):
    """ It's typically more numerically stable *not* to perform the softmax,
    but instead define the loss based on the raw logit predictions. This loss
    function corrects a tensorflow omission where there isn't a KLD loss that
    accepts raw logits. """
    
    # Mask nan values in y_true with zeros
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
    
    return (
            tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True) -
            tf.keras.losses.categorical_crossentropy(y_true, y_true, from_logits=False))


class KLWithLogits(LossFunctionWrapper):
    """ Keras sometimes wants these loss function wrappers to define how to
    reduce the loss over variable batch sizes """
    
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='kl_with_logits'):
        super(KLWithLogits, self).__init__(
            kl_with_logits,
            name=name,
            reduction=reduction)


class PolicyWrapper(layers.Layer):
    
    def build(self, input_shape):
        self.policy_model = policy_model()
    
    def call(self, inputs, mask=None):
        atom, bond, connectivity = inputs
        
        # Get the batch and action dimensions
        atom_shape = tf.shape(atom)
        batch_size = atom_shape[0]
        max_actions_per_node = atom_shape[1]
        
        # Flatten the inputs for running individually through the policy model
        atom_flat = tf.reshape(atom, [batch_size * max_actions_per_node, -1])
        bond_flat = tf.reshape(bond, [batch_size * max_actions_per_node, -1])
        connectivity_flat = tf.reshape(connectivity, [batch_size * max_actions_per_node, -1, 2])
        
        # Get the flat value and prior_logit predictions
        flat_values_logits, flat_prior_logits = self.policy_model([atom_flat, bond_flat, connectivity_flat])
        
        # We put the parent node first in our batch inputs, so this slices
        # the value prediction for the parent
        value_preds = tf.reshape(flat_values_logits, [batch_size, max_actions_per_node, -1])[:, 0, 0]
        
        # Next we get a mask to see where we have valid actions and replace priors for
        # invalid actions with negative infinity (these get zeroed out after softmax).
        # We also only return prior_logits for the child nodes (not the first entry)
        action_mask = tf.reduce_any(tf.not_equal(atom, 0), axis=-1)  # zero is the padding element
        prior_logits = tf.reshape(flat_prior_logits, [batch_size, max_actions_per_node])
        masked_prior_logits = tf.where(action_mask, prior_logits,
                                       tf.ones_like(prior_logits) * prior_logits.dtype.min)[:, 1:]
        
        return value_preds, masked_prior_logits


def build_policy_trainer():
    atom_class = layers.Input(shape=[None, None], dtype=tf.int64, name='atom')
    bond_class = layers.Input(shape=[None, None], dtype=tf.int64, name='bond')
    connectivity = layers.Input(shape=[None, None, 2], dtype=tf.int64, name='connectivity')
    
    value_preds, masked_prior_logits = PolicyWrapper()([atom_class, bond_class, connectivity])
    
    policy_trainer = tf.keras.Model([atom_class, bond_class, connectivity], [value_preds, masked_prior_logits])
    policy_trainer.compile(
        optimizer=tf.keras.optimizers.Adam(config.policy_lr),  # Do AZ list their optimizer?
        loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True), KLWithLogits()])
    
    return policy_trainer
