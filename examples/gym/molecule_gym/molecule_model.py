import tensorflow as tf


class MoleculeModel:
    def __init__(self,
                 policy_model: tf.keras.Model
                 ):
        self.policy_model: tf.keras.Model = policy_model
        self.input, self.value, self.pi = None, None, None
        # self.policy_model_instance = None

        # rlmolecule.molecule.policy.model.policy_model(
        #     features=self.features,
        #     num_heads=self.num_heads,
        #     num_messages=self.num_messages)

    def forward(self, action_observations):
        # self.policy_model_instance = \
        # self.input, self.value, self.pi = self.policy_model(
        #     input_tensors=(action_observations['atom_class'],
        #                    action_observations['bond_class'],
        #                    action_observations['connectivity']))

        # self.value, self.pi = self.policy_model(
        #     [action_observations['atom'],
        #      action_observations['bond'],
        #      action_observations['connectivity']])

        self.value, self.pi = self.policy_model(
            {'atom':action_observations['atom'],
             'bond':action_observations['bond'],
             'connectivity':action_observations['connectivity']})

        # output_tensors = [o.output for o in self.policy_model.outputs]
        return self.value, self.pi
