import math
import random

import numpy as np
import pytest

from rlmolecule.alphazero.alphazero import AlphaZero
from rlmolecule.mcts.mcts import MCTS
from rlmolecule.molecule.molecule_config import MoleculeConfig
from tests.qed_optimization_problem import QEDOptimizationProblem, QEDWithMoleculePolicy


@pytest.fixture
def problem(request, engine):
    name = request.param
    config = MoleculeConfig(max_atoms=4,
                            min_atoms=1,
                            tryEmbedding=False,
                            sa_score_threshold=None,
                            stereoisomers=False)

    if name == 'random':
        return QEDOptimizationProblem(engine, config)
    if name == 'MoleculePolicy':
        return QEDWithMoleculePolicy(engine, config, features=8, num_heads=2, num_messages=1)


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


@pytest.mark.parametrize('solver,problem',
                         [("MCTS", "random"),
                          ("AlphaZero", "random"),
                          ("AlphaZero", "MoleculePolicy")], indirect=True)
class TestMCTSwithMoleculeState:

    def test_reward(self, solver, problem):
        game, root = setup_game(solver, problem)
        assert problem.reward_wrapper(root.state).raw_reward == 0.0

        game._expand(root)

        assert problem.reward_wrapper(root.state).raw_reward == 0.0
        assert problem.reward_wrapper(root.children[-1].state).raw_reward > 0.01
        assert problem.reward_wrapper(root.state).raw_reward == 0.0

    def test_ucb_score(self, solver, problem):
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
            assert game._ucb_score(root, child) == pytest.approx(0.3325546111576978)

    def test_get_successors(self, solver, problem):
        game, root = setup_game(solver, problem)
        game._expand(root)
        children = root.children
        assert len(children) == 9
        assert children[-1].state._forced_terminal
        assert not children[0].state._forced_terminal

    def test_update(self, solver, problem):
        game, root = setup_game(solver, problem)

        root.update(2.)
        assert root.visit_count == 1
        assert root.value == pytest.approx(2.)

        root.update(4.)
        assert root.visit_count == 2
        assert root.value == pytest.approx(3.)

    def test_children(self, solver, problem):
        game, root = setup_game(solver, problem)
        game._expand(root)

        children = root.children
        assert root.children is not None
        assert children[-1].state.forced_terminal
        assert not children[0].state.forced_terminal

        children[0].update(4.)

        assert root.children[0].value == 4.
        assert root.children[0].visit_count == 1

    def test_evaluate(self, solver, problem):
        game, root = setup_game(solver, problem)
        reward = game._evaluate([root])
        assert reward.raw_reward > 0.

    def test_mcts_sample(self, solver, problem):
        random.seed(42)

        game, root = setup_game(solver, problem)
        game.sample(root, 10)

        assert root.visit_count == 10
        assert root.value > 0.1

    def test_run_mcts(self, solver, problem):

        random.seed(42)
        game, root = setup_game(solver, problem)

        history, reward = game.run(num_mcts_samples=5)
        assert len(history) > 1
        assert np.isfinite(reward.raw_reward)

        try:
            assert reward.raw_reward == problem.reward_wrapper(history[-1].state).raw_reward
        except AttributeError:  # Handle alphazero's history object
            assert reward.raw_reward == problem.reward_wrapper(history[-1][0].state).raw_reward
