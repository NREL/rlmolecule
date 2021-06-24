import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel


class GraphGymModel(DistributionalQTFModel):
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
        super(GraphGymModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        self.per_action_model = per_action_model
        self.action_mask = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict['obs']['action_mask']
        action_observations = input_dict['obs']['action_observations']
        batch_size = action_observations[0].shape[0]

        print(f'input_dict {input_dict}')

        flattened_action_features = tf.reshape(
            tf.stack(action_observations, axis=1),
            (batch_size * self.num_actions,) + self.per_action_model.input_shape)

        flat_action_weights = self.per_action_model({'obs': flattened_action_features})[0]
        action_weights = tf.reshape(flat_action_weights, (batch_size, self.num_actions))

        self.action_mask = action_mask
        masked_action_weights = tf.where(action_mask == 0,
                                         action_weights,
                                         tf.ones_like(action_weights) * action_weights.dtype.min)
        return masked_action_weights, state

    def value_function(self):
        inner_vf = self.per_action_model.value_function()
        batch_size = inner_vf.shape[0] // self.num_actions
        action_values = tf.reshape(inner_vf, (batch_size, self.num_actions))
        masked_action_values = tf.where(self.action_mask == 0,
                                        action_values,
                                        tf.ones_like(action_values) * action_values.dtype.min)
        total_value = tf.reduce_max(masked_action_values, axis=1)
        return total_value
