import math
import random

import pytest


@pytest.fixture()
def qed_root():
    from rdkit.Chem.Descriptors import qed
    from rlmolecule.mcts.mcts_node import MCTSNode
    from rlmolecule.molecule.molecule_state import MoleculeState, MoleculeConfig
    import rdkit.Chem

    class QEDNode(MCTSNode):
        def compute_reward(self):
            return qed(self.state.molecule)

    mol = rdkit.Chem.MolFromSmiles('C')
    config = MoleculeConfig(max_atoms=5, min_atoms=1)
    start = MoleculeState(mol, config)

    return QEDNode(start)


def test_reward(qed_root):
    with pytest.raises(AssertionError):
        qed_root.reward

    assert qed_root.get_successors()[-1].reward == 0.3597849378839701


def test_ucb_score(qed_root):
    child = qed_root.get_successors()[0]
    qed_root._visits = 10

    child._visits = 0
    assert qed_root.ucb_score(child) == math.inf

    child._visits = 3
    assert qed_root.ucb_score(child) == pytest.approx(1.7521739232523108)


def test_update(qed_root):
    qed_root.update(2.)
    assert qed_root.visits == 1
    assert qed_root.value == pytest.approx(2.)

    qed_root.update(4.)
    assert qed_root.visits == 2
    assert qed_root.value == pytest.approx(3.)


def test_get_successors(qed_root):
    successors = list(qed_root.get_successors())
    assert len(successors) == 4
    assert successors[-1].terminal
    assert not successors[0].terminal

    successors[0].update(4.)

    assert list(qed_root.get_successors())[0].value == 4.
    assert list(qed_root.get_successors())[0].visits == 1


def test_tree_policy(qed_root):
    assert len(list(qed_root.tree_policy())) == 1

    qed_root.update(1.)
    qed_root.expanded = True
    assert len(list(qed_root.tree_policy())) == 2

    for child in qed_root.get_successors():
        child.update(1.)
        child.expanded = True

    assert len(list(qed_root.tree_policy())) == 3


def test_evaluate(qed_root):
    random.seed(42)
    assert qed_root.evaluate() == pytest.approx(0.4521166661434767)


def test_mcts_step(qed_root):

    random.seed(42)
    for _ in range(10):
        qed_root.mcts_step()

    assert qed_root.visits == 10
    assert qed_root.value == pytest.approx(0.37061839971331995)


def test_run_mcts(qed_root):

    random.seed(42)
    history = list(qed_root.run_mcts(50, explore=False))
    assert history[-1].reward > 0.4