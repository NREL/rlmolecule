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
    config = MoleculeConfig(max_atoms=5, min_atoms=1)
    start = MoleculeState(mol, config)

    return QEDNodeNetworkX(start)

def test_get_successors(qed_nx_root):

    random.seed(42)
    for _ in range(10):
        qed_nx_root.mcts_step()

