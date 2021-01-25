from rlmolecule.mcts.mcts_problem import MCTSProblem
from tests.hallway.hallway_state import HallwayState


class HallwayProblem(MCTSProblem):
    def __init__(self,
                 config: 'HallwayConfig',
                 **kwargs) -> None:
        super(HallwayProblem, self).__init__(**kwargs)
        self._config = config

    def get_initial_state(self) -> HallwayState:
        return HallwayState(1, 0, self._config)

    def get_reward(self, state: HallwayState) -> (float, {}):
        reward = -1.0 * (state.steps + (self._config.size - state.position))
        return reward, {'position': state.position}
