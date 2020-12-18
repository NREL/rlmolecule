import math
import random

import pytest


@pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
def test_reward(qed_case):
    with pytest.raises(AssertionError):
        qed_case.reward

    qed_case.expand()
    assert qed_case.successors[-1].reward == 0.3597849378839701

@pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts'], indirect=True)
def test_ucb_score(qed_case):
    qed_case.expand()
    child = qed_case.successors[0]
    qed_case._visits = 10

    child._visits = 0
    assert qed_case.ucb_score(child) == math.inf

    child._visits = 3
    assert qed_case.ucb_score(child) == pytest.approx(1.7521739232523108)

@pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
def test_update(qed_case):
    qed_case.update(2.)
    assert qed_case.visits == 1
    assert qed_case.value == pytest.approx(2.)

    qed_case.update(4.)
    assert qed_case.visits == 2
    assert qed_case.value == pytest.approx(3.)

@pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
def test_get_successors(qed_case):
    qed_case.expand()
    successors = qed_case.successors
    assert qed_case.expanded
    assert successors[-1].terminal
    assert not successors[0].terminal

    successors[0].update(4.)

    assert qed_case.successors[0].value == 4.
    assert qed_case.successors[0].visits == 1

@pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
def test_tree_policy(qed_case):
    assert len(list(qed_case.tree_policy())) == 1

    qed_case.update(1.)
    qed_case.expand()
    assert len(list(qed_case.tree_policy())) == 2

    for child in qed_case.successors:
        child.update(1.)
        child.expand()

    assert len(list(qed_case.tree_policy())) == 3

@pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
def test_evaluate(qed_case):
    assert qed_case.evaluate() > 0.

@pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
def test_mcts_step(qed_case):
    random.seed(42)
    for _ in range(10):
        qed_case.mcts_step()

    assert qed_case.visits == 10
    assert qed_case.value > 0.1

@pytest.mark.parametrize('qed_case', ['mcts', 'nx_mcts', 'az'], indirect=True)
def test_run_mcts(qed_case):
    random.seed(42)
    history = list(qed_case.run_mcts(5, explore=False))
    assert history[-1].reward > 0.25
