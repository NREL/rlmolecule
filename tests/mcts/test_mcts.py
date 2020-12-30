import math

import pytest

from rlmolecule.molecule.molecule_config import MoleculeConfig
from rlmolecule.alphazero.alphazero import AlphaZero
from rlmolecule.mcts.mcts import MCTS
from tests.qed_optimization_problem import QEDOptimizationProblem


@pytest.fixture
def problem() -> QEDOptimizationProblem:
    config = MoleculeConfig(max_atoms=4,
                            min_atoms=1,
                            tryEmbedding=False,
                            sa_score_threshold=None,
                            stereoisomers=False)
    return QEDOptimizationProblem(config)


@pytest.fixture
def solver(request):
    name = request.param
    if name == 'MCTS':
        return MCTS
    if name == 'AlphaZero':
        return AlphaZero
    raise ValueError('Unknown problem type.')


def setup_game(solver, problem):
    game = solver(problem)
    root = game._get_root()
    return game, root


@pytest.mark.parametrize('solver', ["MCTS", "AlphaZero"], indirect=True)
def test_reward(solver, problem):
    game, root = setup_game(solver, problem)
    assert problem.get_reward(root.state) == 0.0

    game._expand(root)

    assert problem.get_reward(root.state) == 0.0
    assert problem.get_reward(root.children[-1].state) == 0.3597849378839701
    assert problem.get_reward(root.state) == 0.0


@pytest.mark.parametrize('solver', ["MCTS", "AlphaZero"], indirect=True)
def test_ucb_score(solver, problem):
    game, root = setup_game(solver, problem)
    game._expand(root)
    child = root.children[0]

    root.update(2.0)
    assert game._ucb_score(root, child) == math.inf
    assert root.visit_count == 1
    assert root.value == pytest.approx(2.)

    root.update(4.0)
    assert game._ucb_score(root, child) == math.inf
    assert root.visit_count == 2
    assert root.value == pytest.approx(3.)

    child.update(-1.0)
    child.update(0.0)
    if solver is MCTS:
        assert game._ucb_score(root, child) == pytest.approx(0.6774100225154747)


@pytest.mark.parametrize('solver', ["MCTS", "AlphaZero"], indirect=True)
def test_get_successors(solver, problem):
    game, root = setup_game(solver, problem)
    game._expand(root)
    children = root.children
    assert len(children) == 9
    assert children[-1].state._forced_terminal
    assert not children[0].state._forced_terminal


@pytest.mark.parametrize('solver', ["MCTS", "AlphaZero"], indirect=True)
def test_reward(solver, problem):
    game, root = setup_game(solver, problem)
    game._expand(root)
    assert problem.get_reward(root.children[-1].state) == 0.3597849378839701


@pytest.mark.parametrize('solver', ["MCTS", "AlphaZero"], indirect=True)
def test_update(solver, problem):
    game, root = setup_game(solver, problem)

    root.update(2.)
    assert root.visit_count == 1
    assert root.value == pytest.approx(2.)

    root.update(4.)
    assert root.visit_count == 2
    assert root.value == pytest.approx(3.)


@pytest.mark.parametrize('solver', ["MCTS", "AlphaZero"], indirect=True)
def test_children(solver, problem):
    game, root = setup_game(solver, problem)
    game._expand(root)

    children = root.children
    assert root.children is not None
    assert children[-1].state.forced_terminal
    assert not children[0].state.forced_terminal

    children[0].update(4.)

    assert root.children[0].value == 4.
    assert root.children[0].visit_count == 1


@pytest.mark.parametrize('solver', ["MCTS", "AlphaZero"], indirect=True)
def test_evaluate(solver, problem):
    game, root = setup_game(solver, problem)
    assert game._evaluate(root, [root]) > 0.
#

# @pytest.mark.parametrize('solver', ["MCTS", "AlphaZero"], indirect=True)
# def test_tree_policy(solver, problem):
#     game, root = setup_game(solver, problem)
#
#     for i in range(100):
#         assert(len)
#
#     assert len(list(qed_case.tree_policy())) == 1
#
#     qed_case.update(1.)
#     qed_case.expand()
#     assert len(list(qed_case.tree_policy())) == 2
#
#     for child in qed_case.successors:
#         child.update(1.)
#         child.expand()
#
#     assert len(list(qed_case.tree_policy())) == 3


#

# @pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
# def test_mcts_step(qed_case):
#     random.seed(42)
#     for _ in range(10):
#         qed_case.mcts_step()
#
#     assert qed_case.visits == 10
#     assert qed_case.value > 0.1
#
# @pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
# def test_run_mcts(qed_case):
#     random.seed(42)
#     history = list(qed_case.run_mcts(5, explore=False))
#     assert history[-1].reward > 0.25
