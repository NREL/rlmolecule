from rlmolecule.mcts.mcts import MCTS
from rlmolecule.mcts.mcts_vertex import MCTSVertex
from rlmolecule.molecule.builder.builder import MoleculeBuilder
from rlmolecule.tree_search.reward import LinearBoundedRewardFactory
from tests.qed_optimization_problem import QEDWithRandomPolicy


def test_get_successors(engine):
    builder = MoleculeBuilder(max_atoms=4,
                              min_atoms=1,
                              try_embedding=False,
                              sa_score_threshold=None,
                              stereoisomers=False)

    problem = QEDWithRandomPolicy(reward_class=LinearBoundedRewardFactory(), builder=builder, engine=engine)
    game = MCTS(problem)
    root: MCTSVertex = game._get_root()
    game._expand(root)
    root.update(1.0)
    successor0: MCTSVertex = root.children[0]
    game._expand(successor0)
    successor0.update(1.0)

    successor1: MCTSVertex = root.children[1]
    game._expand(successor1)
    successor1.update(1.0)

    CN = root.children[[vertex.state.smiles for vertex in root.children].index('CN')]
    CC = root.children[[vertex.state.smiles for vertex in root.children].index('CC')]

    game._expand(CC)
    game._expand(CN)

    child1 = CC.children[[vertex.state.smiles for vertex in CC.children].index('CCN')]
    child2 = CN.children[[vertex.state.smiles for vertex in CN.children].index('CCN')]

    assert (child1 == child2)
    assert (child1.value == child2.value)
    assert (child1.visit_count == child2.visit_count)

    child1.update(1.)
    assert (child1.value == child2.value)
    assert (child1.visit_count == child2.visit_count)
