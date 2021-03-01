from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils


class PolicyWrapper(layers.Layer):

    def __init__(self,
                 single_position_policy: tf.keras.Model,
                 mask_dict: {str: 'numpy.ndarray'},
                 **kwargs):

        super().__init__(**kwargs)
        self.single_position_policy = single_position_policy
        self.mask_dict = mask_dict

    def call(self, inputs, mask=None):

        # print("INPUTS", inputs)

        # Get the batch and action dimensions
        shape = tf.shape(inputs[0])
        batch_size = shape[0]
        max_actions_per_node = shape[1]
        flattened_batch = batch_size * max_actions_per_node
        # print(shape, batch_size, max_actions_per_node, flattened_shape)

        # Flatten the inputs for running individually through the policy model
        flattened_inputs = []
        for inp in inputs:
            inp_shape = tf.shape(inp)
            flattened_shape = [flattened_batch, *[inp_shape[k] for k in range(2, len(inp_shape))]]
            flattened_inputs += [tf.reshape(inp, flattened_shape)]

        # Get the flat value and prior_logit predictions
        # print("FLATTENED_INPUTS", flattened_inputs)
        flat_values_logits, flat_prior_logits = self.single_position_policy(flattened_inputs)

        # We put the parent node first in our batch inputs, so this slices
        # the value prediction for the parent
        value_preds = tf.reshape(flat_values_logits, [batch_size, max_actions_per_node, -1])[:, 0, 0]
        prior_logits = tf.reshape(flat_prior_logits, [batch_size, max_actions_per_node])

        # Next we get a mask to see where we have valid actions and replace priors for
        # invalid actions with negative infinity (these get zeroed out after softmax).
        # We also only return prior_logits for the child nodes (not the first entry).
        # todo: use the input mask instead of making our own masking layer
        action_mask = tf.cast(tf.ones_like(prior_logits), tf.bool)
        for i, inp in enumerate(inputs):
            inp = tf.reshape(inp, [batch_size, max_actions_per_node, -1])
            new_mask = tf.reduce_all(
                tf.not_equal(inp, self.mask_dict[self.single_position_policy.inputs[i].name]),
                axis=-1)
            action_mask = tf.logical_and(action_mask, new_mask)

        # Apply the mask
        masked_prior_logits = tf.where(
            action_mask,
            prior_logits,
            tf.ones_like(prior_logits) * prior_logits.dtype.min)[:, 1:]

        return value_preds, masked_prior_logits

    @classmethod
    def build_policy_model(cls,
                           single_position_policy: tf.keras.Model,
                           mask_dict: {str: 'numpy.ndarray'}
                           ) -> tf.keras.Model:
        """ Main entry point for initializing the wrapped policy model. Expands the input dimensions of the single
        position policy model in order to account for batches of positions.

        :param single_position_policy: A tf.keras.Model instance that produces
        a policy prediction from a single batch of actions
        :param mask_dict: A dictionary of padding values that represent masked actions
        :return: An initilized policy model that acts over batches of (parent, children) inputs
        """

        inputs = single_position_policy.inputs
        input_names = list(mask_dict.keys())
        assert len(inputs) == len(input_names), \
            "Mismatch in number of inputs between policy model and policy inputs mask"

        batched_inputs = []
        batched_inputs_with_mask = []

        for single_input in inputs:
            for input_name in input_names:
                if single_input.name.startswith(input_name):
                    break
            else:
                raise RuntimeError(f"Input with name {single_input.name} not found in policy inputs")

            batched_input = tf.keras.Input(shape=single_input.shape,
                                           name=input_name,
                                           dtype=single_input.dtype)
            batched_inputs += [batched_input]
            batched_inputs_with_mask += [layers.Masking(mask_dict[input_name])(batched_input)]

        value_preds, masked_prior_logits = cls(single_position_policy, mask_dict)(batched_inputs_with_mask)
        return tf.keras.Model(batched_inputs, [value_preds, masked_prior_logits])


class TimeCsvLogger(tf.keras.callbacks.CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        super(TimeCsvLogger, self).on_epoch_end(epoch, logs)


def kl_with_logits(y_true, y_pred) -> tf.Tensor:
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


def get_input_mask_dict(inputs: list,
                        mask_dict: dict = {},
                        as_tensor: bool = False,
                        value: float = 0.) -> dict:
    """Returns a dictionary of mask values with type cast to that of the
    corresponding input layer."""
    _mask_dict = {inp.name: value for inp in inputs}
    _mask_dict.update(mask_dict)
    if as_tensor:
        return {inp.name: tf.constant(_mask_dict[inp.name], dtype=inp.dtype) for inp in inputs}
    else:
        return {inp.name: _mask_dict[inp.name] for inp in inputs}
