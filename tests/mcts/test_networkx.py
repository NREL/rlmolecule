import pytest
import random

@pytest.fixture()
def qed_nx_root():
    from rdkit.Chem.Descriptors import qed
    from rlmolecule.mcts.mcts_node import MCTSNode
    from rlmolecule.mcts.networkx_node import NetworkxSuccessorMixin
    from rlmolecule.molecule.molecule_state import MoleculeState, MoleculeConfig
    import rdkit.Chem

    class QEDNodeNetworkX(NetworkxSuccessorMixin, MCTSNode):
        def compute_reward(self):
            return qed(self.state.molecule)

    mol = rdkit.Chem.MolFromSmiles('C')
    config = MoleculeConfig(max_atoms=4,
                            min_atoms=1,
                            tryEmbedding=False,
                            sa_score_threshold=None,
                            stereoisomers=False)

    start = MoleculeState(mol, config)

    return QEDNodeNetworkX(start)

def test_get_successors(qed_nx_root):

    qed_nx_root.expand().update(1.)
    qed_nx_root.successors[1].expand().update(1.)
    qed_nx_root.successors[0].expand().update(1.)

    random.seed(42)
    for _ in range(2):
        qed_nx_root.mcts_step()

    child1 = qed_nx_root.successors[1].successors[0]  # CCN
    child2 = qed_nx_root.successors[0].successors[1]  # CCN

    assert child1 == child2
    assert child1.value == child2.value
    assert child1.visits == child2.visits

    child1.update(1.)
    assert child1.value == child2.value
    assert child1.visits == child2.visits