from typing import Optional

import nfp
from graphenv import tf
from nfp.preprocessing import MolPreprocessor
from rlmolecule.policy.preprocessor import load_preprocessor

layers = tf.keras.layers


def policy_model(
    preprocessor: Optional[MolPreprocessor] = None,
    features: int = 64,
    num_messages: int = 3,
    input_dtype: str = "int64",
    max_atoms: Optional[int] = None,
    max_bonds: Optional[int] = None,
) -> tf.keras.Model:
    """Constructs a policy model that predicts value, pi_logits from a batch of molecule
    inputs. Main model used in policy training and loading weights

    :param preprocessor: a MolPreprocessor class for initializing the embedding matrices
    :param features: Size of network hidden layers
    :param num_messages: Number of message passing layers
    :param input_dtype: the datatype of the input arrays
    :param max_atoms: the shape of the input atom_class layer
    :return: The constructed policy model
    """
    if preprocessor is None:
        preprocessor = load_preprocessor()

    # Define inputs
    atom_class = layers.Input(
        shape=[max_atoms], dtype=input_dtype, name="atom"
    )  # batch_size, num_atoms
    bond_class = layers.Input(
        shape=[max_bonds], dtype=input_dtype, name="bond"
    )  # batch_size, num_bonds
    connectivity = layers.Input(
        shape=[max_bonds, 2], dtype=input_dtype, name="connectivity"
    )  # batch_size, num_bonds, 2

    input_tensors = [atom_class, bond_class, connectivity]

    # Initialize the atom states
    atom_state = layers.Embedding(
        preprocessor.atom_classes, features, name="atom_embedding", mask_zero=True
    )(atom_class)

    # Initialize the bond states
    bond_state = layers.Embedding(
        preprocessor.bond_classes, features, name="bond_embedding", mask_zero=True
    )(bond_class)

    for _ in range(num_messages):  # Do the message passing
        new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
        atom_state = layers.Add()([atom_state, new_atom_state])

    global_state = layers.GlobalAveragePooling1D()(atom_state)
    value_logit = layers.Dense(1)(global_state)
    pi_logit = layers.Dense(1)(global_state)

    return tf.keras.Model(input_tensors, [value_logit, pi_logit], name="policy_model")
