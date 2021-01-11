import random
from unittest.mock import MagicMock

import pytest

from rlmolecule.alphazero.alphazero import AlphaZero
from rlmolecule.molecule.molecule_config import MoleculeConfig
from tests.qed_optimization_problem import QEDWithMoleculePolicy


@pytest.fixture
def problem(engine):
    config = MoleculeConfig(max_atoms=4,
                            min_atoms=1,
                            tryEmbedding=False,
                            sa_score_threshold=None,
                            stereoisomers=False)

    return QEDWithMoleculePolicy(engine, config, features=8, num_heads=2, num_messages=1)


@pytest.fixture
def game(problem):
    game = AlphaZero(problem)
    return game


def test_reward_caching(game):
    root = game._get_root()
    problem = game.problem

    game.problem.get_reward = MagicMock(return_value=(1, {}))

    reward1 = game.problem._reward_wrapper(root.state)
    reward2 = game.problem._reward_wrapper(root.state)

    assert reward1 == reward2
    assert game.problem.get_reward.call_count == 1


class TestPolicyTraining:

    @pytest.mark.parametrize('execution_number', range(5))
    def test_create_games(self, game, execution_number):
        random.seed(42)
        history, reward = game.run(num_mcts_samples=5)
        from rlmolecule.sql.tables import Game
        stored_game = game.problem.session.query(Game).filter_by(id=str(game.id)).one()
        assert stored_game.scaled_reward == reward

    def test_recent_games(self, problem):
        recent_games = list(problem.iter_recent_games())
        assert len(recent_games) == 5

    def test_policy_data(self, problem):
        data = problem.create_dataset()
        list(data.take(1))
    # outputs = list(game_iterator(engine))
    # outputs[0]
