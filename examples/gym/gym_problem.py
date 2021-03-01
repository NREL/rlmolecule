from copy import deepcopy

import sqlalchemy
import tensorflow as tf

from rlmolecule.alphazero.tfalphazero_problem import TFAlphaZeroProblem
from alphazero_gym import AlphaZeroGymEnv
from gym_state import GymEnvState


class GymEnvProblem(TFAlphaZeroProblem):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 model: tf.keras.Model,
                 env: AlphaZeroGymEnv,
                 **kwargs) -> None:
        super().__init__(engine, model, **kwargs)
        self._env = deepcopy(env)

    def get_initial_state(self) -> GymEnvState:
        _ = self._env.reset()
        return GymEnvState(self._env, 0., 0., False)

    def get_reward(self, state: GymEnvState) -> (float, dict):
        return state._env.cumulative_reward, {}

    def get_policy_inputs(self, state: GymEnvState) -> dict:
        return {"obs": self._env.get_obs()}
