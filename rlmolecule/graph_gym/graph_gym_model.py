import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel

from examples.gym.molecule_gym.molecule_model import MoleculeModel


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

        self.per_action_model = MoleculeModel(per_action_model())
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
        # print(f'flat_action_observations {flat_action_observations["atom"].shape} {flat_action_observations["bond"].shape} {flat_action_observations["connectivity"].shape}', flush=True)
        flat_action_values, flat_action_weights = tuple(self.per_action_model.forward(flat_action_observations))

        # reform action values and weights from [v(b0,a0), v(b0,a1), ..., v(b1,a0), ...] into
        # [ [v(b0,a0), v(b0,a1), ...], [b(b1,a0), ...], ...]
        # and set invalid actions to the minimum value

        # print(f'flat_action_values {flat_action_values.shape} flat_action_weights {flat_action_weights.shape}', flush=True)
        # print(f'flat_action_values {flat_action_values}')
        # print(f'action_mask {action_mask}')
        action_values = tf.reshape(flat_action_values, action_mask_shape)
        action_values = tf.where(action_mask, action_values, action_values.dtype.min)
        self.total_value = tf.reduce_max(action_values, axis=1)

        action_weights = tf.reshape(flat_action_weights, action_mask_shape)
        action_weights = tf.where(action_mask, action_weights, action_weights.dtype.min)

        # print(f'action_values {action_values}\naction_weights {action_weights}')
        # print(f'action_values {action_values.shape}\naction_weights {action_weights.shape}')
        return action_weights, state

    # def forward(self, input_dict, state, seq_lens):
    #     # Extract the available actions tensor from the observation.
    #     action_mask = input_dict['obs']['action_mask']
    #     if action_mask.dtype != tf.dtypes.bool:
    #         action_mask = tf.equal(action_mask, 1.0)
    #     action_observations = input_dict['obs']['action_observations']
    #
    #     # TODO: might need to extend this code to deal with non-dict shaped observation spaces
    #
    #     # evaluate each action observation tensor
    #     evaluations = [tuple(self.per_action_model.forward(action_observation))
    #                    for action_observation in action_observations]
    #
    #     def reduce_and_mask(index):
    #         t = tf.stack([e[index] for e in evaluations], axis=-2)
    #         t = tf.reshape(t, t.shape[0:-1])
    #         t = tf.where(action_mask, t, t.dtype.min)
    #         return t
    #
    #     action_values = reduce_and_mask(0)
    #     action_weights = reduce_and_mask(1)
    #
    #     # print(f'model values action_mask {action_mask.shape} len(evaluations) {len(evaluations)} '
    #     #       f'evaluations[0] {evaluations[0].shape} evaluations[1] {evaluations[1].shape} '
    #     #       f'action_values {action_values.shape} action_weights {action_weights.shape}')
    #     self.total_value = tf.reduce_max(action_values, axis=1)
    #     return action_weights, state

    def value_function(self):
        return self.total_value
