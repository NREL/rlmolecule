from copy import deepcopy
from typing import Tuple

import sqlalchemy

from rlmolecule.alphazero.tfalphazero_problem import TFAlphaZeroProblem

from examples.gym.alphazero_gym import AlphaZeroGymEnv
from examples.gym.gym_state import GymEnvState


class GymEnvProblem(TFAlphaZeroProblem):
    """Gym env TF AZ problem that automates the parent class abstractmethods 
    via the gym interface and gym state."""
    
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 env: AlphaZeroGymEnv,
                 **kwargs) -> None:
        self.env = deepcopy(env)
        super().__init__(engine, **kwargs)
        
    def get_initial_state(self) -> GymEnvState:
        _ = self.env.reset()
        return GymEnvState(self.env, 0., 0., False)



