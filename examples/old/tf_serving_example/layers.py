# Some draft layers, I'd like to eventually add these to the nfp repo

import nfp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers.pooling import GlobalPooling1D
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class ConcatDense(layers.Layer):
    """ Layer to combine the concatenation and two dense layers """

    def build(self, input_shape):
        num_features = input_shape[0][-1]
        self.concat = layers.Concatenate()
        self.dense1 = layers.Dense(2 * num_features, activation='relu')
        self.dense2 = layers.Dense(num_features)

    def call(self, inputs, mask=None):
        output = self.concat(inputs)
        output = self.dense1(output)
        output = self.dense2(output)
        return output


class GraphLayer(layers.Layer):
    """ Base class for all GNN layers """

    def __init__(self, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout

    def build(self, input_shape):
        if len(input_shape) == 4:
            self.use_global = True
            self.tile = Tile()

        elif len(input_shape) == 3:
            self.use_global = False

        else:
            raise RuntimeError("wrong input shape")

        if self.dropout > 0.:
            self.dropout_layer = layers.Dropout(self.dropout)

    def get_config(self):
        return {"dropout": self.dropout}


class EdgeUpdate(GraphLayer):
    def build(self, input_shape):
        """ inputs = [atom_state, bond_state, connectivity]
        shape(bond_state) = [batch, num_bonds, bond_features]
        """
        super().build(input_shape)

        bond_features = input_shape[1][-1]

        self.gather = nfp.Gather()
        self.slice1 = nfp.Slice(np.s_[:, :, 1])
        self.slice0 = nfp.Slice(np.s_[:, :, 0])

        self.concat = ConcatDense()
        self.add = layers.Add()

    def call(self, inputs, mask=None):
        """ Inputs: [atom_state, bond_state, connectivity]
            Outputs: bond_state
        """
        if not self.use_global:
            atom_state, bond_state, connectivity = inputs
        else:
            atom_state, bond_state, connectivity, global_state = inputs
            global_state = self.tile([global_state, bond_state])

        # Get nodes at start and end of edge
        source_atom = self.gather([atom_state, self.slice1(connectivity)])
        target_atom = self.gather([atom_state, self.slice0(connectivity)])

        if not self.use_global:
            new_bond_state = self.concat([bond_state, source_atom, target_atom])
        else:
            new_bond_state = self.concat([bond_state, source_atom, target_atom, global_state])

        if self.dropout > 0.:
            new_bond_state = self.dropout_layer(new_bond_state)

        new_bond_state = self.add([bond_state, new_bond_state])
        return new_bond_state

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class NodeUpdate(GraphLayer):
    def build(self, input_shape):
        super().build(input_shape)

        num_features = input_shape[1][-1]

        self.gather = nfp.Gather()
        self.slice0 = nfp.Slice(np.s_[:, :, 0])
        self.slice1 = nfp.Slice(np.s_[:, :, 1])

        self.concat = ConcatDense()
        self.reduce = nfp.Reduce(reduction='sum')

        self.dense1 = layers.Dense(2 * num_features, activation='relu')
        self.dense2 = layers.Dense(num_features)
        self.add = layers.Add()

    def call(self, inputs, mask=None):
        """ Inputs: [atom_state, bond_state, connectivity]
            Outputs: atom_state
        """
        if not self.use_global:
            atom_state, bond_state, connectivity = inputs
        else:
            atom_state, bond_state, connectivity, global_state = inputs
            global_state = self.tile([global_state, bond_state])

        source_atom = self.gather([atom_state, self.slice1(connectivity)])

        if not self.use_global:
            messages = self.concat([source_atom, bond_state])
        else:
            messages = self.concat([source_atom, bond_state, global_state])

        new_atom_state = self.reduce([messages, self.slice0(connectivity), atom_state])

        # Dense net after message reduction
        new_atom_state = self.dense1(new_atom_state)
        new_atom_state = self.dense2(new_atom_state)

        if self.dropout > 0.:
            new_atom_state = self.dropout_layer(new_atom_state)

        new_atom_state = self.add([atom_state, new_atom_state])

        return new_atom_state

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Tile(layers.Layer):
    def call(self, inputs):
        global_state, target = inputs
        target_shape = tf.shape(target)[1]  # number of edges or nodes
        expanded = tf.expand_dims(global_state, 1)
        return tf.tile(expanded, tf.stack([1, target_shape, 1]))


class GlobalUpdate(GraphLayer):
    def __init__(self, units, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.units = units  # H
        self.num_heads = num_heads  # N

    def build(self, input_shape):
        super().build(input_shape)
        dense_units = self.units * self.num_heads  # N*H
        self.query_layer = layers.Dense(self.num_heads, name='query')
        self.value_layer = layers.Dense(dense_units, name='value')
        self.add = layers.Add()

    def transpose_scores(self, input_tensor):
        input_shape = tf.shape(input_tensor)
        output_shape = [input_shape[0], input_shape[1], self.num_heads, self.units]
        output_tensor = tf.reshape(input_tensor, output_shape)
        return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,S,H]

    def call(self, inputs, mask=None):

        if not self.use_global:
            atom_state, bond_state, connectivity = inputs
        else:
            atom_state, bond_state, connectivity, global_state = inputs

        batch_size = tf.shape(atom_state)[0]

        graph_elements = tf.concat([atom_state, bond_state], axis=1)
        query = self.query_layer(graph_elements)  # [B,N,S,H]
        query = tf.transpose(query, perm=[0, 2, 1])
        value = self.transpose_scores(self.value_layer(graph_elements))  # [B,N,S,H]

        attention_probs = tf.nn.softmax(query)
        context = tf.matmul(tf.expand_dims(attention_probs, 2), value)
        context = tf.reshape(context, [batch_size, self.num_heads * self.units])

        if self.dropout > 0.:
            context = self.dropout_layer(context)

        if self.use_global:
            global_state = self.add([global_state, context])
        else:
            global_state = context

        return global_state

    def get_config(self):
        config = super(GlobalUpdate, self).get_config()
        config.update({"units": self.units, "num_heads": self.num_heads})
        return config


class GlobalSumPooling1D(GlobalPooling1D):
    def __init__(self, data_format='channels_last', **kwargs):
        super().__init__(data_format=data_format, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == 'channels_last' else 2
        if mask is not None:
            mask = math_ops.cast(mask, backend.floatx())
            mask = array_ops.expand_dims(mask, 2 if self.data_format == 'channels_last' else 1)

            inputs *= mask

        return backend.sum(inputs, axis=steps_axis)

    def compute_mask(self, inputs, mask=None):
        return None
