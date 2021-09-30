import tensorflow as tf


class MoleculeModel:
    def __init__(self,
                 policy_model: tf.keras.Model
                 ):
        self.policy_model: tf.keras.Model = policy_model
        self.input, self.value, self.pi = None, None, None

    def forward(self, action_observations):
       
        self.value, self.pi = self.policy_model(
            {'fingerprint': action_observations['fingerprint']}
        )
        return self.value, self.pi
