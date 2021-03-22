from copy import deepcopy

from rlmolecule.tree_search.reward import Reward
from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.gym.alphazero_gym import AlphaZeroGymEnv
from rlmolecule.gym.gym_state import GymEnvState


class GymProblem(MCTSProblem):
    def __init__(self, *, reward_class: Reward, env: AlphaZeroGymEnv, **kwargs):
        self.env = deepcopy(env)
        super().__init__(reward_class=reward_class, **kwargs)

    def get_initial_state(self) -> GymEnvState:
        _ = self.env.reset()
        return GymEnvState(self.env, 0, 0., 0., False)

        