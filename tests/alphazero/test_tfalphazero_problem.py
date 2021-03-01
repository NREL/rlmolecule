"""DB: Not spending time here recalling how to use pytest etc.  Let's get the
basic functionality working and we can improve.
TODO:  What are the right tests here for multiple model types / input features?"""

import unittest

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from rlmolecule.alphazero.tf_keras_policy import PolicyWrapper, get_input_mask_dict


# Example function for building a simple model
def policy_model() -> tf.keras.Model:

    inp1 = layers.Input([None, 1], dtype=tf.float32, name="input1")
    inp2 = layers.Input([None, 1], dtype=tf.float32, name="input2")

    # Concatenate inputs
    x = layers.Concatenate()((inp1, inp2))

    # Dense layers and concatenation
    x = layers.Dense(4, activation="relu")(x)
    x = layers.Dense(4, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    value_logit = layers.Dense(1, name="value")(x)
    pi_logit = layers.Dense(1, name="prior")(x)

    return tf.keras.Model([inp1, inp2], [value_logit, pi_logit], name="policy_model")


class TestPolicyWrapperAndHelpers(unittest.TestCase):
        
    def test_mask_maker(self):
        model = policy_model()
        print(model.outputs)
        mask_dict = get_input_mask_dict(model.inputs, {}, as_tensor=False)
        assert np.all(list(mask_dict.values()) == [0.] * 2)


    def test_policy_wrapper(self):

        model = policy_model()
        wrapper = PolicyWrapper(model)
        input_keys = [inp.name for inp in model.inputs]
        assert all([k in input_keys for k in wrapper.mask_dict])
        assert all([v == 0. for v in wrapper.mask_dict.values()])

        wrapper = PolicyWrapper(model, {model.inputs[1].name: -1.})
        assert all([k in input_keys for k in wrapper.mask_dict])
        assert wrapper.mask_dict[model.inputs[1].name] == -1.


if __name__ == "__main__":
    unittest.main()
