import tensorflow as tf
from ray.rllib.models.tf import TFModelV2


class GraphGymModel(TFModelV2):
    """
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 per_action_model):
        super(GraphGymModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        # print(f"obs_space {obs_space}")
        # print(f"obs_space.original_space {obs_space.original_space}")
        # print(f"action_space {action_space}")
        # print(f"num_outputs {num_outputs}")

        self.per_action_model = per_action_model
        # self.total_value = None
        self.action_values = None
        # observation_space = obs_space.original_space

    def forward(self, input_dict, state, seq_lens):
        # print(f'forward() {state}')
        # print(f'input_dict {input_dict}')
        # print(f"input_dict['obs'] {input_dict['obs']}")
        # print(f"input_dict['obs']['action_observations'] {input_dict['obs']['action_observations']}")

        # Extract the available actions tensor from the observation.
        observation = input_dict['obs']
        action_mask = observation['action_mask']

        if action_mask.dtype != tf.dtypes.bool:
            action_mask = tf.equal(action_mask, 1.0)
        action_observations = observation['action_observations']
        state_observation = observation['state_observation']

        action_mask_shape = tf.shape(action_mask)  # batch size, num_actions

        flat_observations = {}
        for key in state_observation.keys():
            action_observations_sublist = [state_observation[key]] \
                                          + [action_observation[key] for action_observation in action_observations]
            stacked_observations = tf.stack(action_observations_sublist, axis=1)
            stacked_shape = tf.shape(stacked_observations)  # batch size, feature sizes ...
            flat_shape = tf.concat([tf.reshape(stacked_shape[0] * stacked_shape[1], (1,)), stacked_shape[2:]], axis=0)
            flat_observations[key] = tf.reshape(stacked_observations, flat_shape)

        # run flattened action observations through the per action model to evaluate each action
        # flat_values, flat_weights = tuple(self.per_action_model.forward(flat_observations))
        flat_values, flat_weights = tuple(self.per_action_model(flat_observations))
        composite_shape = tf.stack([action_mask_shape[0], action_mask_shape[1] + 1], axis=0)
        action_weights = tf.where(action_mask,
                                  tf.reshape(flat_weights, composite_shape)[:, 1:],
                                  flat_weights.dtype.min)

        self.action_values = tf.reshape(flat_values, composite_shape)[:, 0]

        # print(
        #     f"action_mask {tf.shape(action_mask)} {action_mask}\n"
        #     f"action_weights {tf.shape(action_weights)} {action_weights}\n{flat_weights}\n"
        #     f"action_values {tf.shape(self.action_values)} {self.action_values}\n{flat_values}\n")
        return action_weights, state

    def value_function(self):
        return self.action_values
