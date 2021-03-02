"""DB: Not spending time here recalling how to use pytest etc.  Let's get the
basic functionality working and we can improve.
TODO:  What are the right tests here for multiple model types / input features?"""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

from rlmolecule.alphazero.tf_keras_policy import PolicyWrapper, align_input_names


# Example function for building a simple model


@pytest.fixture(scope='function')
def model() -> tf.keras.Model:
    inp1 = layers.Input([1], dtype=tf.float32, name="input1")
    inp2 = layers.Input([None, 2], dtype=tf.float32, name="input2")

    inp2_reduced = layers.GlobalAveragePooling1D()(inp2)
    # Concatenate inputs
    x = layers.Concatenate()((inp1, inp2_reduced))

    # Dense layers and concatenation
    x = layers.Dense(4, activation="relu")(x)
    x = layers.Dense(4, activation="relu")(x)

    value_logit = layers.Dense(1, name="value")(x)
    pi_logit = layers.Dense(1, name="prior")(x)

    return tf.keras.Model([inp1, inp2], [value_logit, pi_logit], name="policy_model")


def test_policy_wrapper(model):
    input_mask = {'input1': -1., 'input2': -2.}
    wrapper = PolicyWrapper.build_policy_model(model, input_mask)
    input_names = align_input_names(wrapper.inputs, input_mask)
    assert input_names == ['input1', 'input2']
    assert wrapper.inputs[0].shape.as_list() == [None, None, 1]
    assert wrapper.inputs[1].shape.as_list() == [None, None, None, 2]

    assert wrapper.layers[2].mask_value == -1.
    assert wrapper.layers[3].mask_value == -2.

    inputs1 = np.random.randn(10, 1)
    inputs2 = np.random.randn(10, 5, 2)
    inputs1_masked = np.concatenate([inputs1, -1. * np.ones((3, 1))])
    inputs2_masked = np.concatenate([inputs2, -2. * np.ones((3, 5, 2))])

    model_output = model([inputs1, inputs2])
    wrapped_output = wrapper([inputs1[np.newaxis, :, :], inputs2[np.newaxis, :, :, :]])
    wrapped_output_masked = wrapper([inputs1_masked[np.newaxis, :, :], inputs2_masked[np.newaxis, :, :, :]])

    assert model_output[0][0].numpy().flatten() == wrapped_output[0][0].numpy().flatten()  # parent values line up
    assert np.allclose(model_output[1].numpy().flatten()[1:], wrapped_output[1].numpy().flatten())  # priors match

    assert wrapped_output[0][0].numpy().flatten() == wrapped_output_masked[0][0].numpy().flatten()
    assert np.allclose(wrapped_output[1].numpy().flatten(),
                       wrapped_output_masked[1].numpy().flatten()[:9])  # priors match
    assert np.all(wrapped_output_masked[1].numpy().flatten()[9:] < -1E10)  # should be essentially -inf
