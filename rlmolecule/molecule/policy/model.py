from typing import Optional

import nfp
import tensorflow as tf
from nfp.preprocessing.mol_preprocessor import MolPreprocessor
from tensorflow.keras import layers

from rlmolecule.molecule.policy.preprocessor import load_preprocessor


def policy_model(preprocessor: Optional[MolPreprocessor] = None,
                 features: int = 64,
                 num_heads: int = 4,
                 num_messages: int = 3,
                 input_dtype: str = 'int64') -> tf.keras.Model:
    """ Constructs a policy model that predicts value, pi_logits from a batch of molecule inputs. Main model used in
    policy training and loading weights

    :param preprocessor: a MolPreprocessor class for initializing the embedding matrices
    :param features: Size of network hidden layers
    :param num_heads: Number of global state attention heads. Must be a factor of `features`
    :param num_messages: Number of message passing layers
    :param input_dtype: the datatype of the input arrays
    :return: The constructed policy model
    """
    if preprocessor is None:
        preprocessor = load_preprocessor()

    # Define inputs
    atom_class = layers.Input(shape=[None], dtype=input_dtype, name='atom')  # batch_size, num_atoms
    bond_class = layers.Input(shape=[None], dtype=input_dtype, name='bond')  # batch_size, num_bonds
    connectivity = layers.Input(shape=[None, 2], dtype=input_dtype, name='connectivity')  # batch_size, num_bonds, 2

    input_tensors = [atom_class, bond_class, connectivity]

    # Initialize the atom states
    atom_state = layers.Embedding(preprocessor.atom_classes, features, name='atom_embedding',
                                  mask_zero=True)(atom_class)

    # Initialize the bond states
    bond_state = layers.Embedding(preprocessor.bond_classes, features, name='bond_embedding',
                                  mask_zero=True)(bond_class)

    units = features // num_heads
    global_state = nfp.GlobalUpdate(units=units, num_heads=num_heads)([atom_state, bond_state, connectivity])

    for _ in range(num_messages):  # Do the message passing
        new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
        atom_state = layers.Add()([atom_state, new_atom_state])

        new_global_state = nfp.GlobalUpdate(units=units,
                                            num_heads=num_heads)([atom_state, bond_state, connectivity, global_state])
        global_state = layers.Add()([global_state, new_global_state])

    value_logit = layers.Dense(1)(global_state)
    pi_logit = layers.Dense(1)(global_state)

    return tf.keras.Model(input_tensors, [value_logit, pi_logit], name='policy_model')
