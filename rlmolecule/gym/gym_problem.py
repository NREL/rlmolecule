from copy import deepcopy

import sqlalchemy

from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.alphazero.tfalphazero_problem import TFAlphaZeroProblem
from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.gym.alphazero_gym import AlphaZeroGymEnv
from rlmolecule.gym.gym_state import GymEnvState


class MCTSGymProblem(MCTSProblem):
    """Gym env MCTS problem that automates the parent class abstractmethods 
    via the gym interface and gym state."""
    
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 env: AlphaZeroGymEnv,
                 **kwargs) -> None:
        self.env = deepcopy(env)
        super().__init__(engine=engine, **kwargs)

    def get_initial_state(self) -> GymEnvState:
        _ = self.env.reset()
        return GymEnvState(self.env, 0, 0., 0., False)


class TFAlphaZeroGymProblem(MCTSGymProblem, AlphaZeroProblem):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 env: AlphaZeroGymEnv,
                 **kwargs) -> None:
        super().__init__(engine=engine, env=env, **kwargs)


class TFAlphaZeroGymProblem(MCTSGymProblem, TFAlphaZeroProblem):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 env: AlphaZeroGymEnv,
                 **kwargs) -> None:
        super().__init__(engine=engine, env=env, **kwargs)

