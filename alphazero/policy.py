import tensorflow as tf
from tensorflow.keras import layers
import nfp

import alphazero.config as config
from alphazero.preprocessor import preprocessor

# two models: 
# first, a policy model that predicts value, pi_logits from a batch of molecule inputs
# Then, a wrapper model that expects batches of games and normalizes logit values

def policy_model():
       
    # Define inputs
    atom_class = layers.Input(shape=[None], dtype=tf.int64, name='atom') # batch_size, num_atoms
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond') # batch_size, num_bonds
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity') # batch_size, num_bonds, 2
    
    input_tensors = [atom_class, bond_class, connectivity]

    # Initialize the atom states
    atom_state = layers.Embedding(preprocessor.atom_classes, config.features,
                                  name='atom_embedding', mask_zero=True)(atom_class)

    # Initialize the bond states
    bond_state = layers.Embedding(preprocessor.bond_classes, config.features,
                                  name='bond_embedding', mask_zero=True)(bond_class)
    
    units = config.features//config.num_heads
    global_state = nfp.GlobalUpdate(units=units, num_heads=config.num_heads)(
        [atom_state, bond_state, connectivity])

    for _ in range(config.num_messages):  # Do the message passing
        bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
        atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
        global_state = nfp.GlobalUpdate(units=units, num_heads=config.num_heads)(
            [atom_state, bond_state, connectivity, global_state])
        
    value = layers.Dense(1, activation='tanh')(global_state)
    pi_logit = layers.Dense(1)(global_state)
    
    return tf.keras.Model(input_tensors, [value, pi_logit], name='policy_model')
