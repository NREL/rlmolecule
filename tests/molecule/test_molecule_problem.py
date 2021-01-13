import numpy as np
import pytest
from rdkit.Chem.rdmolfiles import MolFromSmiles

from rlmolecule.alphazero.alphazero import AlphaZero
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.molecule.molecule_config import MoleculeConfig
from rlmolecule.molecule.molecule_state import MoleculeState
from tests.qed_optimization_problem import QEDWithMoleculePolicy


@pytest.fixture
def config():
    return MoleculeConfig(max_atoms=4,
                          min_atoms=1,
                          tryEmbedding=False,
                          sa_score_threshold=None,
                          stereoisomers=False)


@pytest.fixture
def problem(engine, config):
    return QEDWithMoleculePolicy(engine, config)


@pytest.fixture
def vertex(config):
    return AlphaZeroVertex(MoleculeState(MolFromSmiles('CCC'), config))


@pytest.fixture
def solver(problem):
    return AlphaZero(problem)


def test_get_network_inputs(problem, vertex):
    network_inputs = problem._get_network_inputs(vertex.state)
    assert len(network_inputs['atom']) == vertex.state.num_atoms
    assert len(network_inputs['bond']) == 2 * vertex.state.molecule.GetNumBonds()


def test_get_batched_network_inputs(solver, vertex):
    solver._expand(vertex)
    batched_network_inputs = solver.problem._get_batched_network_inputs(vertex)
    assert batched_network_inputs['atom'].shape[0] == len(vertex.children) + 1
    assert batched_network_inputs['atom'].ndim == 2
    assert batched_network_inputs['connectivity'].ndim == 3

    assert (batched_network_inputs['atom'][5] ==
            solver.problem._get_network_inputs(vertex.children[4].state)['atom']).all()


def test_get_value_and_policy(solver, vertex):
    solver._expand(vertex)
    value, priors = solver.problem.get_value_and_policy(vertex)

    assert np.isfinite(value)
    assert np.isclose(sum(priors.values()), 1.)
