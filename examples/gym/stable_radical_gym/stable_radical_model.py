import operator
from functools import reduce

import rlmolecule
from rlmolecule.graph_gym.graph_gym_env import GraphGymEnv


class StableRadicalModel:
    def __init__(self,
                 graph_gym: GraphGymEnv,
                 features: int = 64,
                 num_heads: int = 4,
                 num_messages: int = 3,
                 ):
        self.graph_gym: GraphGymEnv = graph_gym

        self.features: int = features
        self.num_heads: int = num_heads
        self.num_messages: int = num_messages

        # self.input_shape: (int, ...) = \
        #     (sum((reduce(operator.mul, (s.shape), 1) for s in self.graph_gym.observation_space.values())),)


    def do_thing(self):
        keras_model = rlmolecule.molecule.policy.model.policy_model(
            features=self.features,
            num_heads=self.num_heads,
            num_messages=self.num_messages)

        keras_model.inputs
        pass
