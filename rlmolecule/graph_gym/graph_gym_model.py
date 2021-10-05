import time
import traceback

import numpy as np
import tensorflow as tf
#from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

#TODO (Dave): Don't hardcode to new example!
from examples.gym.molecule_gym.molecule_model import MoleculeModel
#from examples.gym.mol_fp_gym.molecule_model import MoleculeModel


class GraphGymModel(TFModelV2):
    """
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        super(GraphGymModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        self.action_embed_model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs,
            model_config, name + "_action_embed")

    # def forward(self, input_dict, state, seq_lens):
    #     start = time.perf_counter()

    #     # Extract the available actions tensor from the observation.
    #     action_mask = input_dict['obs']['action_mask']
    #     if action_mask.dtype != tf.dtypes.bool:
    #         action_mask = tf.equal(action_mask, 1.0)
    #     action_observations = input_dict['obs']['action_observations']

    #     #(f'action_mask {action_mask.shape}')
    #     # print(f'input_dict {input_dict}')
    #     action_mask_shape = tf.shape(action_mask)  # batch size, num_actions
    #     # TODO: might need to extend this code to deal with non-dict shaped observation spaces

    #     # print(f'action_observations {action_observations}')

    #     # flatten action observations into a single dict with tensors stacked like:
    #     # [(batch 0, action 0), (b0,a1), ..., (b1,a0), ...]
    #     #flat_action_observations = {}
    #     #for key in action_observations[0].keys():
    #     action_observations_sublist = [action_observation["fingerprint"] for action_observation in action_observations]
    #     flat_action_observations = tf.concat(action_observations_sublist, axis=0)

    #     # run flattened action observations through the per action model to evaluate each action
    #     #print(f'flat_action_observations {flat_action_observations["fingerprint"].shape}')
    #     flat_action_values, flat_action_weights = tuple(self.action_embed_model(
    #         {"obs": flat_action_observations}))

    #     # reform action values and weights from [v(b0,a0), v(b0,a1), ..., v(b1,a0), ...] into
    #     # [ [v(b0,a0), v(b0,a1), ...], [b(b1,a0), ...], ...]
    #     # and set invalid actions to the minimum value

    #     # print(f'flat_action_values {flat_action_values.shape} flat_action_weights {flat_action_weights.shape}', flush=True)
    #     # print(f'flat_action_values {flat_action_values}')
    #     # print(f'flat_action_weights {flat_action_weights}')
    #     # print(f'action_mask {action_mask}')
    #     action_values = tf.reshape(flat_action_values, action_mask_shape)
    #     action_values = tf.where(action_mask, action_values, action_values.dtype.min)
    #     self.total_value = tf.reduce_max(action_values, axis=1)

    #     action_weights = tf.reshape(flat_action_weights, action_mask_shape)
    #     action_weights = tf.where(action_mask, action_weights, action_weights.dtype.min)

    #     # print(f'action_values {action_values}\naction_weights {action_weights}')
    #     # print(f'action_values {action_values.shape}\naction_weights {action_weights.shape}')

    #     #print(f'GraphGymModel::forward() {(time.perf_counter() - start) * 1000}')
    #     # try:
    #     #     raise TypeError("Oups!")
    #     # except Exception:
    #     #     traceback.print_stack(limit=40)

    #     # try:
    #     #     norm = sum([np.linalg.norm(w) for w in self.per_action_model.policy_model.get_weights()])
    #     #     print("total_weights_norm", norm)
    #     # except:
    #     #     pass

    #     return action_weights, state

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        avail_actions = input_dict["obs"]["action_observations"]
        action_mask = input_dict["obs"]["action_mask"]

        # LAST THOUGHT FOR THE NIGHT:  They compute an embedding for the current
        # state, then take inner product with actions to get the logits.  We 
        # just want to compute the logits for each next state (independent of 
        # current state).
        # Compute the predicted action embedding
        print(input_dict.keys())
        action_embed, _ = self.action_embed_model({
            "obs": input_dict["fingerprint"]
        })

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = tf.expand_dims(action_embed, 1)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=2)

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


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
        # print("value", self.total_value)
        #return self.total_value
        return self.action_embed_model.value_function()

