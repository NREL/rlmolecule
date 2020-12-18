import math

import pytest

# def test_reward(qed_root):
# @pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
from molecule_game.molecule_config import MoleculeConfig
from rlmolecule.mcts.mcts_game import MCTSGame
from tests.qed_optimization_problem import QEDOptimizationProblem

@pytest.fixture()
def game() -> MCTSGame:
    config = MoleculeConfig(max_atoms=4,
                            min_atoms=1,
                            tryEmbedding=False,
                            sa_score_threshold=None,
                            stereoisomers=False)
    problem = QEDOptimizationProblem(config)
    game = MCTSGame(problem)
    return game


def test_reward(game):
    root = game._make_root()
    assert game.compute_reward(root) == 0.0

    game._expand(root)

    assert game.compute_reward(root) == 0.0
    assert game.compute_reward(root.children[-1]) == 0.3597849378839701
    assert game.compute_reward(root) == 0.0


def test_ucb_score(game):
    root = game._make_root()
    game._expand(root)
    child = root.children[0]

    root.update(2.0)
    assert game._ucb_score(root, child) == math.inf
    assert root.visits == 1
    assert root.value == pytest.approx(2.)

    root.update(4.0)
    assert game._ucb_score(root, child) == math.inf
    assert root.visits == 2
    assert root.value == pytest.approx(3.)

    game._backpropagate([root, child], 3.0)
    assert game._ucb_score(root, child) == pytest.approx(5.09629414793641)


def test_get_successors(game):
    root = game._make_root()
    successors = game._expand(root)
    assert len(successors) == 9
    assert successors[-1].state.is_terminal
    assert not successors[0].state.is_terminal

# @pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
# def test_tree_policy(qed_case):
#     assert len(list(qed_case.tree_policy())) == 1
#
# def test_tree_policy(qed_root):
#     assert len(list(qed_root.tree_policy())) == 1
#     qed_case.update(1.)
#     qed_case.expand()
#     assert len(list(qed_case.tree_policy())) == 2
#
#     qed_root.update(1.)
#     qed_root.expand()
#     assert len(list(qed_root.tree_policy())) == 2
#
#     for child in qed_root.successors:
#     for child in qed_case.successors:
#         child.update(1.)
#         child.expand()
#
#     assert len(list(qed_root.tree_policy())) == 3
#     assert len(list(qed_case.tree_policy())) == 3
#
# @pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
# def test_evaluate(qed_case):
#     assert qed_case.evaluate() > 0.
#
# def test_evaluate(qed_root):
#     random.seed(42)
#     assert qed_root.evaluate() == pytest.approx(0.4521166661434767)
#
#
# def test_mcts_step(qed_root):
# @pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
# def test_mcts_step(qed_case):
#     random.seed(42)
#     for _ in range(10):
#         qed_root.mcts_step()
#         qed_case.mcts_step()
#
#     assert qed_root.visits == 10
#     assert qed_root.value == pytest.approx(0.37061839971331995)
#     assert qed_case.visits == 10
#     assert qed_case.value > 0.1
#
#
# def test_run_mcts(qed_root):
# @pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
# def test_run_mcts(qed_case):
#     random.seed(42)
#     history = list(qed_root.run_mcts(10, explore=False))
#     assert history[-1].reward > 0.4
#     history = list(qed_case.run_mcts(5, explore=False))
#     assert history[-1].reward > 0.3
#
