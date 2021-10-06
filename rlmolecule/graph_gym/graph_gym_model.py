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
        # self.total_value = None
        self.action_values = None

    def forward(self, input_dict, state, seq_lens):
        # print(f'input_dict {input_dict}')
        # print(f"input_dict['obs'] {input_dict['obs']}")
        # print(f"input_dict['obs']['action_observations'] {input_dict['obs']['action_observations']}")

        # Extract the available actions tensor from the observation.
        observation = input_dict['obs']
        action_mask = observation['action_mask']

        if tf.reduce_sum(action_mask) == 0:
            print(f"terminal forward {state}")

        if action_mask.dtype != tf.dtypes.bool:
            action_mask = tf.equal(action_mask, 1.0)
        action_observations = observation['action_observations']
        state_observation = observation['state_observation']

        action_mask_shape = tf.shape(action_mask)  # batch size, num_actions

        # flatten action observations into a single dict with tensors stacked like:
        # [(batch 0, action 0), (b0,a1), ..., (b1,a0), ...]
        # [(batch 0, action 0), (b1,a0), ..., (b0,a1), ...]
        flat_observations = {}
        for key in state_observation.keys():
            action_observations_sublist = [state_observation[key]] \
                                          + [action_observation[key] for action_observation in action_observations]

            # [(b0, a0), (b1, a0)...], [(b0, a1), (b1, a1), ...], ...
            # batch size * num actions, num features per action
            stacked_observations = tf.stack(action_observations_sublist, axis=1)
            stacked_shape = tf.shape(stacked_observations)  # batch size, feature sizes ...
            flat_shape = tf.concat([tf.reshape(stacked_shape[0] * stacked_shape[1], (1,)), stacked_shape[2:]], axis=0)
            # print(f'key {key}, observation_shape {observation_shape} stacked_shape {stacked_shape}'
            #       f' {flat_shape} {flat_shape.shape}')
            flat_observations[key] = tf.reshape(stacked_observations, flat_shape)

        # run flattened action observations through the per action model to evaluate each action
        flat_values, flat_weights = tuple(self.per_action_model.forward(flat_observations))

        # [(batch 0, action 0), (b1,a0), ..., (b0,a1), ...]
        # desired: batch, action  (0,0), (0, 1), ... (1,0), (1,1)...

        # reform action values and weights from [v(b0,a0), v(b0,a1), ..., v(b1,a0), ...] into
        # [ [v(b0,a0), v(b0,a1), ...], [b(b1,a0), ...], ...]
        # and set invalid actions to the minimum value
        composite_shape = tf.stack([action_mask_shape[0], action_mask_shape[1] + 1], axis=0)
        action_weights = tf.where(action_mask,
                                  tf.reshape(flat_weights, composite_shape)[:, 1:],
                                  flat_weights.dtype.min)

        # self.action_values = tf.where(action_mask,
        #                               tf.reshape(flat_values, action_mask_shape),
        #                               flat_values.dtype.min)

        self.action_values = tf.reshape(flat_values, composite_shape)[:, 0]

        print(
            f"action_weights {tf.shape(action_weights)} action_values {tf.shape(self.action_values)} {self.action_values}")

        # action_values = tf.reshape(flat_action_values, action_mask_shape)
        # self.action_values = tf.where(action_mask, action_values, action_values.dtype.min)
        # self.action_values = tf.where(action_mask, action_values, tf.constant(0.0))
        # self.action_values = tf.maximum(self.action_values, tf.constant(0.0))
        # self.total_value = tf.reduce_max(action_values, axis=1)

        # action_weights = tf.reshape(flat_action_weights, action_mask_shape)
        # action_weights = tf.where(action_mask, action_weights, action_weights.dtype.min)
        # return action_weights, state
        return action_weights, state

    def value_function(self):
        # return self.total_value
        # return tf.maximum(tf.constant(0.0), tf.reduce_max(self.action_values, axis=1))
        # return tf.reduce_max(self.action_values, axis=1) * .99
        return self.action_values
