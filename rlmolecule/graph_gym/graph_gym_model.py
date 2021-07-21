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
        self.total_value = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict['obs']['action_mask']
        if action_mask.dtype != tf.dtypes.bool:
            action_mask = tf.equal(action_mask, 1.0)
        action_observations = input_dict['obs']['action_observations']

        # print(f'action_mask {action_mask.shape}')
        # print(f'input_dict {input_dict}')
        action_mask_shape = tf.shape(action_mask)  # batch size, num_actions
        # TODO: might need to extend this code to deal with non-dict shaped observation spaces

        # print(f'action_observations {action_observations}')

        # flatten action observations into a single dict with tensors stacked like:
        # [(batch 0, action 0), (b0,a1), ..., (b1,a0), ...]
        flat_action_observations = {}
        for key in action_observations[0].keys():
            action_observations_sublist = [action_observation[key] for action_observation in action_observations]
            flat_action_observations[key] = tf.concat(action_observations_sublist, axis=0)

        # run flattened action observations through the per action model to evaluate each action
        # print(f'flat_action_observations {flat_action_observations}')
        flat_action_values, flat_action_weights = tuple(self.per_action_model.forward(flat_action_observations))

        # reform action values and weights from [v(b0,a0), v(b0,a1), ..., v(b1,a0), ...] into
        # [ [v(b0,a0), v(b0,a1), ...], [b(b1,a0), ...], ...]
        # and set invalid actions to the minimum value

        # print(f'flat_action_values {flat_action_values}')
        # print(f'action_mask {action_mask}')
        action_values = tf.reshape(flat_action_values, action_mask_shape)
        action_values = tf.where(action_mask, action_values, action_values.dtype.min)
        self.total_value = tf.reduce_max(action_values, axis=1)

        action_weights = tf.reshape(flat_action_weights, action_mask_shape)
        action_weights = tf.where(action_mask, action_weights, action_weights.dtype.min)

        # print(f'action_values {action_values}\naction_weights {action_weights}')
        return action_weights, state

    def value_function(self):
        return self.total_value
