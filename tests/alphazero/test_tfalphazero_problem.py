"""DB: Not spending time here recalling how to use pytest etc.  Let's get the
basic functionality working and we can improve."""

import unittest

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from rlmolecule.alphazero.tfalphazero_problem import (
    PolicyWrapper, make_input_mask, make_action_mask, flatten_batch_and_action_axes)


# Example function for building a simple model
def policy_model() -> tf.keras.Model:

    # Position input
    position = layers.Input([None, 1], dtype=tf.float32, name="position")

    # Steps input
    steps = layers.Input([None, 1], dtype=tf.float32, name="steps")

    # Concatenate inputs
    x = layers.Concatenate()((position, steps))

    # Dense layers and concatenation
    x = layers.Dense(4, activation="relu")(x)
    x = layers.Dense(4, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    value_logit = layers.Dense(1, name="value")(x)
    pi_logit = layers.Dense(1, name="prior")(x)

    return tf.keras.Model([position, steps], [value_logit, pi_logit], name="policy_model")



class TestPolicyWrapperAndHelpers(unittest.TestCase):

    def test_mask_values(self):
        
        mask_value = -1

        inp = np.ones((2, 1, 1))
        inp[1, 0, 0] = -1  # should mask the child

        answer = tf.constant([[True], [False]], dtype=bool)
        assert tf.reduce_all(make_input_mask(tf.constant(inp), mask_value) == answer)

        
    def test_mask_inputs(self):

        input_feature_dims = [1, 3]
        input_masks = [-1, 0]

        batch_size = 3
        max_actions_per_node = 2

        inp1 = np.ones((batch_size, max_actions_per_node, input_feature_dims[0]))
        inp1[0, 0, 0] = input_masks[0]

        inp2 = np.ones((batch_size, max_actions_per_node, input_feature_dims[1]))
        inp2[1, 1, 1] = input_masks[1]
        
        inputs = [tf.constant(inp1), tf.constant(inp2)]
        result = make_action_mask(inputs, input_masks)
        
        answer = tf.constant(
            [[False, True],
             [True, False],
             [True, True]], dtype=bool)

        assert tf.reduce_all(result == answer)

    def test_flatten_along_batch_and_action(self):
        shape = [5, 4, 3, 2]
        inp = tf.ones(shape)
        ans = tf.ones((shape[0]*shape[1], shape[2], shape[3]))
        assert tf.reduce_all(flatten_batch_and_action_axes(inp) == ans)

    def test_policy_wrapper(self):

        model = policy_model()
        wrapper = PolicyWrapper(model)
        input_keys = [inp.name for inp in model.inputs]
        assert all([k in input_keys for k in wrapper.input_masks])
        assert all([v == 0. for v in wrapper.input_masks.values()])

        wrapper = PolicyWrapper(model, {model.inputs[1].name: -1.})
        assert all([k in input_keys for k in wrapper.input_masks])
        assert wrapper.input_masks[model.inputs[1].name] == -1.


if __name__ == "__main__":
    unittest.main()