import pytest

from examples.hallway.hallway_config import HallwayConfig
from rlmolecule.mcts.mcts import MCTS
from rlmolecule.tree_search.reward import LinearBoundedRewardFactory
from tests.hallway.hallway_problem import HallwayProblem


@pytest.fixture
def game():
    config = HallwayConfig(size=16, max_steps=16)
    reward_factory = LinearBoundedRewardFactory(min_reward=-30, max_reward=-15)
    return MCTS(HallwayProblem(config, reward_class=reward_factory))


class TestMCTSHallway:

    def test_run(self, game):
        vertex = game._get_root()
        game.sample(vertex, 10000)
        # todo, actually write tests to make sure that this solves the optimal hallway. Also write AZ hallway tests
        result = game.run(num_mcts_samples=500)