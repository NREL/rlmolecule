import nfp
import numpy as np
import rdkit
import rdkit.Chem

import tensorflow as tf
from tensorflow.keras import layers

def atom_featurizer(atom):
    """ Return an integer hash representing the atom type
    """

    return str((
        atom.GetSymbol(),
        atom.GetNumRadicalElectrons(),
        atom.GetFormalCharge(),
        atom.GetChiralTag(),
        atom.GetIsAromatic(),
        nfp.get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True)
    ))


def bond_featurizer(bond, flipped=False):
    
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))
    
    btype = str(bond.GetBondType())
    ring = 'R{}'.format(nfp.get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
    
    return " ".join([atoms, btype, ring]).strip()


class MolPreprocessor(nfp.preprocessing.SmilesPreprocessor):
    def construct_feature_matrices(self, mol, train=True):
        """ Convert an rdkit Mol to a list of tensors
        'atom' : (n_atom,) length list of atom classes
        'bond' : (n_bond,) list of bond classes
        'connectivity' : (n_bond, 2) array of source atom, target atom pairs.
        """
        
        self.atom_tokenizer.train = train
        self.bond_tokenizer.train = train
        
        if self.explicit_hs:
            mol = rdkit.Chem.AddHs(mol)

        n_atom = mol.GetNumAtoms()
        n_bond = 2 * mol.GetNumBonds()

        # If its an isolated atom, add a self-link
        if n_bond == 0:
            n_bond = 1
        
        atom_feature_matrix = np.zeros(n_atom, dtype='int')
        bond_feature_matrix = np.zeros(n_bond, dtype='int')
        connectivity = np.zeros((n_bond, 2), dtype='int')

        bond_index = 0
        for n, atom in enumerate(mol.GetAtoms()):

            # Atom Classes
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom))

            start_index = atom.GetIdx()

            for bond in atom.GetBonds():
                # Is the bond pointing at the target atom
                rev = bond.GetBeginAtomIdx() != start_index

                # Bond Classes
                bond_feature_matrix[bond_index] = self.bond_tokenizer(
                    self.bond_features(bond, flipped=rev))         

                # Connectivity
                if not rev:  # Original direction
                    connectivity[bond_index, 0] = bond.GetBeginAtomIdx()
                    connectivity[bond_index, 1] = bond.GetEndAtomIdx()

                else:  # Reversed
                    connectivity[bond_index, 0] = bond.GetEndAtomIdx()
                    connectivity[bond_index, 1] = bond.GetBeginAtomIdx()

                bond_index += 1

        return {
            'atom': atom_feature_matrix,
            'bond': bond_feature_matrix,
            'connectivity': connectivity,
        }
    

preprocessor = MolPreprocessor(atom_features=atom_featurizer,
                               bond_features=bond_featurizer,
                               explicit_hs=False)


class Tile(layers.Layer):
    def __init__(self, axis, multiple, *args, **kwargs):
        """ Expands the input layer on `axis` with tf.expand_dims,
        followed by a tf.tile using `multiple`. Should probably
        include this in a future version of nfp.
        """
        super(Tile, self).__init__(*args, **kwargs)
        self.axis = axis
        self.multiple = multiple
        
    def build(self, input_shape):
        tile_shape = [1] * len(input_shape)
        tile_shape.insert(self.axis, self.multiple)
        self.tile_shape = tf.constant(tile_shape)
    
    def call(self, inputs):
        expanded = tf.expand_dims(inputs, self.axis)
        return tf.tile(expanded, self.tile_shape)


def build_gnn_model(preprocessor):
    """ Builds the GNN model with keras/tensorflow """
    
    atom_features = 32
    num_messages = 3
    
    # Define inputs
    atom_class = layers.Input(shape=[None], dtype=tf.int64, name='atom')
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

    input_tensors = [atom_class, bond_class, connectivity]

    # Initialize the atom states
    atom_state = layers.Embedding(preprocessor.atom_classes, atom_features,
                                  name='atom_embedding', mask_zero=True)(atom_class)

    # Initialize the bond states
    bond_state = layers.Embedding(preprocessor.bond_classes, atom_features,
                                  name='bond_embedding', mask_zero=True)(bond_class)
    

    def message_block(original_atom_state,
                      original_bond_state,
                      connectivity):
        """ Performs the graph-aware updates """

        atom_state = layers.LayerNormalization()(original_atom_state)
        bond_state = layers.LayerNormalization()(original_bond_state)

        source_atom = nfp.Gather()([atom_state, nfp.Slice(np.s_[:, :, 1])(connectivity)])
        target_atom = nfp.Gather()([atom_state, nfp.Slice(np.s_[:, :, 0])(connectivity)])
        
        # Edge update network
        new_bond_state = layers.Concatenate()(
            [source_atom, target_atom, bond_state])
        new_bond_state = layers.Dense(
            2*atom_features, activation='relu')(new_bond_state)
        new_bond_state = layers.Dense(atom_features)(new_bond_state)

        bond_state = layers.Add()([original_bond_state, new_bond_state])

        # message function
        source_atom = layers.Dense(atom_features)(source_atom)    
        messages = layers.Multiply()([source_atom, bond_state])
        messages = nfp.Reduce(reduction='sum')(
            [messages, nfp.Slice(np.s_[:, :, 0])(connectivity), atom_state])

        # state transition function
        messages = layers.Dense(atom_features, activation='relu')(messages)
        messages = layers.Dense(atom_features)(messages)

        atom_state = layers.Add()([original_atom_state, messages])

        return atom_state, bond_state, 

    for i in range(num_messages):
        atom_state, bond_state = message_block(atom_state, bond_state, connectivity)

    # Here, I'm predicting a single value for each molecule.
    # So I pool over all the atoms and reduce the global state to a single number
    # as the output. For a policy network, this might be two vectors (a value and probability_logit)
    global_state = layers.GlobalAveragePooling1D()(atom_state)
    output = layers.Dense(1)(global_state)

    model = tf.keras.Model(input_tensors, output)
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(1E-3))
    
    return model