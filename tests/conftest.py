import pytest

@pytest.fixture(scope='function')
def molecule_start():
    import rdkit.Chem
    from rlmolecule.molecule.molecule_state import MoleculeState, MoleculeConfig

    mol = rdkit.Chem.MolFromSmiles('C')
    config = MoleculeConfig(max_atoms=4,
                            min_atoms=1,
                            tryEmbedding=False,
                            sa_score_threshold=None,
                            stereoisomers=False)

    start = MoleculeState(mol, config)

    return start


@pytest.fixture(scope='function')
def QedNode():
    from rdkit.Chem.Descriptors import qed

    class QedNode:
        def compute_reward(self):
            return qed(self.state.molecule)

    return QedNode

@pytest.fixture(scope='function')
def qed_mcts(QedNode, molecule_start):
    from rlmolecule.mcts.mcts_node import MCTSNode

    class QEDMcts(QedNode, MCTSNode):
        pass

    return QEDMcts(molecule_start)


@pytest.fixture(scope='function')
def qed_nx_mcts(QedNode, molecule_start):
    from rlmolecule.mcts.mcts_node import MCTSNode
    from rlmolecule.mcts.networkx_node import NetworkxSuccessorMixin

    class QEDNodeNetworkX(QedNode, NetworkxSuccessorMixin, MCTSNode):
        pass

    return QEDNodeNetworkX(molecule_start)


@pytest.fixture(scope='function')
def qed_az(QedNode, molecule_start):
    from rlmolecule.alphazero.alphazero_node import AlphaZeroNode
    from rlmolecule.alphazero.alphazero_game import AlphaZeroGame
    from rlmolecule.alphazero.test import RandomPolicy

    class QEDNodeAz(QedNode, RandomPolicy, AlphaZeroNode):
        pass

    game = AlphaZeroGame()

    return QEDNodeAz(molecule_start, game)


@pytest.fixture(scope='function')
def qed_case(request, qed_mcts, qed_nx_mcts, qed_az):
    type = request.param
    if type == 'mcts':
        return qed_mcts
    elif type == 'nx_mcts':
        return qed_nx_mcts
    elif type == 'az':
        return qed_az
    else:
        raise ValueError('unknown type')