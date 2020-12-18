import pytest
import random

def test_get_successors(qed_nx_mcts):

    qed_nx_mcts.expand().update(1.)
    qed_nx_mcts.children[1].expand().update(1.)
    qed_nx_mcts.children[0].expand().update(1.)

    random.seed(42)
    for _ in range(2):
        qed_nx_mcts.mcts_step()

    child1 = qed_nx_mcts.children[1].children[0]  # CCN
    child2 = qed_nx_mcts.children[0].children[1]  # CCN

    assert child1 == child2
    assert child1.value == child2.value
    assert child1.visits == child2.visits

    child1.update(1.)
    assert child1.value == child2.value
    assert child1.visits == child2.visits