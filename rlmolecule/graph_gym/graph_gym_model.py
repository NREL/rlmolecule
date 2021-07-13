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
        action_observations = input_dict['obs']['action_observations']

        print(f'action_mask {action_mask.shape}')
        print(f'input_dict {input_dict}')
        batch_size, num_actions = action_mask.shape
        action_mask_shape = tf.shape(action_mask)  # batch size, num_actions
        # num_actions = action_mask.shape[1]

        # TODO: might need to extend this code to deal with non-dict shaped observation spaces

        # action_observations is a list of dicts, one for each successor/action.
        # take each action's observations and compile them into a single dict of tensors

        # print(f'action_observations {action_observations}')
        # flat_action_observations = {}
        # for key in action_observations[0].keys():
        #     action_observations_sublist = [action_observation[key] for action_observation in action_observations]
        #     flat_action_observations[key] = tf.concat(action_observations_sublist, axis=0)
        #
        # print(f'combined_action_observations {flat_action_observations}')
        # flat_action_values, flat_action_weights = tuple(self.per_action_model.forward(flat_action_observations))
        #
        #
        #
        # print(f'flat_action_values {flat_action_values}')
        # action_values = tf.reshape(flat_action_values, action_mask_shape)
        # action_values = tf.reshape(action_values, action_mask.shape)
        # action_values = tf.where(action_mask == 0,
        #                          action_values,
        #                          tf.ones_like(action_values) * action_values.dtype.min)
        # self.total_value = tf.reduce_max(action_values, axis=1)
        #
        # action_weights = tf.reshape(flat_action_weights, action_mask_shape)
        # masked_action_weights = tf.where(action_mask == 0,
        #                                  action_weights,
        #                                  tf.ones_like(action_weights) * action_weights.dtype.min)
        # return masked_action_weights, state

        # action_outputs = \
        #     [tuple(self.per_action_model.forward(action_observation)) for action_observation in action_observations]
        # action_values = tf.stack([tf.reshape(a[0], action_mask.shape) for a in action_outputs], axis=1)
        # action_weights = tf.stack([tf.reshape(a[1], action_mask.shape) for a in action_outputs], axis=1)

        action_model = self.per_action_model.forward(action_observation)

        print(f'action_values {action_values}')
        print(f'action_weights {action_weights}')

        action_values = tf.where(action_mask == 0,
                                 action_values,
                                 tf.ones_like(action_values) * action_values.dtype.min)
        self.total_value = tf.reduce_max(action_values, axis=1)

        masked_action_weights = tf.where(action_mask == 0,
                                         action_weights,
                                         tf.ones_like(action_weights) * action_weights.dtype.min)
        return masked_action_weights, state

    def value_function(self):
        return self.total_value
