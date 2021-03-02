from copy import deepcopy
from typing import Tuple

import sqlalchemy

from rlmolecule.alphazero.tfalphazero_problem import TFAlphaZeroProblem

from alphazero_gym import AlphaZeroGymEnv
from gym_state import GymEnvState


class GymEnvProblem(TFAlphaZeroProblem):
    """Gym env TF AZ problem that automates the parent class abstractmethods 
    via the gym interface and gym state."""
    
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 env: AlphaZeroGymEnv,
                 **kwargs) -> None:
        self._env = deepcopy(env)
        super().__init__(engine, **kwargs)

    def get_initial_state(self) -> GymEnvState:
        _ = self._env.reset()
        return GymEnvState(self._env, 0., 0., False)

    def get_reward(self, state: GymEnvState) -> Tuple[float, dict]:
        return state.cumulative_reward, {}

