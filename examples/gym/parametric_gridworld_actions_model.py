import copy

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf import TFModelV2, FullyConnectedNetwork


class ParametricGridworldActionsModel(DistributionalQTFModel):
    """Parametric action model that handles the dot product and masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 example_observation_space={},
                 # true_obs_shape=(4,),
                 # action_embed_size=2,
                 **kw):
        super(ParametricGridworldActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        # self.action_embed_model = FullyConnectedNetwork(
        #     gym.spaces.Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size,
        #     model_config, name + '_action_embed')
        # print(f'init actions: {example_observation_space["action_observations"]}')
        action_observations_space = example_observation_space['action_observations']
        action_observation_space = action_observations_space[0]
        self.per_action_model = FullyConnectedNetwork(
            action_observation_space, action_space, 1,
            model_config, name + '_per_action_model')
        self.num_actions = len(action_observations_space)
        self.action_feature_shape = action_observation_space.shape
        self.action_mask = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict['obs']['action_mask']
        action_observations = input_dict['obs']['action_observations']

        # print(f'forward: {action_mask.shape} {len(action_observations)} {action_observations[0].shape}')

        batch_size = action_observations[0].shape[0]
        # action_feature_shape = action_observations[0].shape[1:]

        flattened_action_features = tf.reshape(
            tf.stack(action_observations, axis=1),
            (batch_size * self.num_actions, ) + self.action_feature_shape)

        flat_action_weights = self.per_action_model({'obs': flattened_action_features})[0]
        #
        #
        # print(f'stacked {stacked.shape}')
        # action_weights_outputs = \
        #     [self.per_action_model.base_model({'obs': action_observation})[0]
        #      for action_observation in action_observations]
        action_weights = tf.reshape(flat_action_weights, (batch_size, self.num_actions))

        # print(f'action_weights {action_weights.shape}  action_mask {action_mask.shape}')

        # Mask out invalid actions (use tf.float32.min for stability)
        # TODO: this seems like not a good way to mask
        # inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        self.action_mask = action_mask
        masked_action_weights = tf.where(action_mask != 0, action_weights,
                                       tf.ones_like(action_weights) * action_weights.dtype.min)
        return masked_action_weights, state

    def value_function(self):
        # print('value_function')

        inner_vf = self.per_action_model.value_function()
        # print(f'inner_vf {inner_vf.shape}')
        batch_size = inner_vf.shape[0] // self.num_actions
        action_values = tf.reshape(inner_vf, (batch_size, self.num_actions))
        masked_action_values = tf.where(self.action_mask != 0, action_values,
                                         tf.ones_like(action_values) * action_values.dtype.min)
        total_value = tf.reduce_max(action_values, axis=1)
        # print(f'total_value {total_value.shape}')
        return total_value
