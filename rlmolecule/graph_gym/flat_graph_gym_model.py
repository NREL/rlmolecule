import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel

from examples.gym.molecule_gym.molecule_model import MoleculeModel


class FlatGraphGymModel(DistributionalQTFModel):
    """
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 per_action_model,
                 **kw):
        super(FlatGraphGymModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        self.per_action_model = MoleculeModel(per_action_model(
            # features=8, num_heads=1, num_messages=1
        ))
        self.total_value = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        observation = input_dict['obs']
        action_mask = observation['action_mask']
        if action_mask.dtype != tf.dtypes.bool:
            action_mask = tf.equal(action_mask, 1.0)

        action_mask_shape = tf.shape(action_mask)  # batch size, num_actions
        # TODO: might need to extend this code to deal with non-dict shaped observation spaces

        # flatten action observations into a single dict with tensors stacked like:
        # [(batch 0, action 0), (b0,a1), ..., (b1,a0), ...])
        flat_batch_size = action_mask_shape[0] * action_mask_shape[1]
        flat_action_observations = {
            key: tf.reshape(value,
                            (flat_batch_size, tf.shape(value)[1] // action_mask_shape[1], *value.shape[2:]))
            for key, value in observation.items() if key != 'action_mask'}

        # run flattened action observations through the per action model to evaluate each action
        flat_action_values, flat_action_weights = tuple(self.per_action_model.forward(flat_action_observations))

        # reform action values and weights from [v(b0,a0), v(b0,a1), ..., v(b1,a0), ...] into
        # [ [v(b0,a0), v(b0,a1), ...], [b(b1,a0), ...], ...]
        # and set invalid actions to the minimum value

        action_values = tf.reshape(flat_action_values, action_mask_shape)
        action_values = tf.where(action_mask, action_values, action_values.dtype.min)
        self.total_value = tf.reduce_max(action_values, axis=1)

        action_weights = tf.reshape(flat_action_weights, action_mask_shape)
        action_weights = tf.where(action_mask, action_weights, action_weights.dtype.min)
        return action_weights, state

    def value_function(self):
        return self.total_value
