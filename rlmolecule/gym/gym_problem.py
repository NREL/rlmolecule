from copy import deepcopy
from typing import Tuple

from rlmolecule.gym.alphazero_gym import AlphaZeroGymEnv
from rlmolecule.gym.gym_state import GymEnvState
from rlmolecule.mcts.mcts_problem import MCTSProblem


class GymProblem(MCTSProblem):
    def __init__(self,
                 env: AlphaZeroGymEnv,
                 **kwargs):
        self.env = deepcopy(env)
        super().__init__(**kwargs)

    def get_initial_state(self) -> GymEnvState:
        _ = self.env.reset()
        return GymEnvState(self.env, 0, 0., 0., False)

    def get_reward(self, state: GymEnvState) -> Tuple[float, dict]:
        return state.cumulative_reward, {}
