import numpy as np
import pytest
import rdkit
from graphenv.graph_env import GraphEnv
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.qed import QEDState
from rlmolecule.molecule_state import MoleculeState


@pytest.fixture
def propane(builder: MoleculeBuilder) -> MoleculeState:
    return MoleculeState(
        rdkit.Chem.MolFromSmiles("CCC"), builder=builder, force_terminal=False
    )


def test_root(propane: MoleculeState):
    root = propane.root
    assert root.smiles == "C"


def test_next_actions(propane: MoleculeState):
    next_actions = propane.children
    butanes = list(filter(lambda x: x.smiles == "CCCC", next_actions))
    assert len(butanes) == 1
    assert butanes[0].forced_terminal is False
    assert next_actions[-1].forced_terminal is True


def test_prune_terminal(builder):

    qed_root = QEDState(
        rdkit.Chem.MolFromSmiles("C"),
        builder,
        smiles="C",
        max_num_actions=20,
        prune_terminal_states=True,
    )

    env = GraphEnv({"state": qed_root, "max_num_children": qed_root.max_num_actions})
    assert repr(env.state.children[-1]) == "C (t)"

    # select the terminal state
    obs, reward, terminal, info = env.step(len(env.state.children) - 1)
    assert terminal
    assert np.isclose(reward, 0.3597849378839701)

    obs = env.reset()
    obs, reward, terminal, info = env.step(len(env.state.children) - 1)
    assert not terminal
    assert np.isclose(reward, 0)


def test_prune_terminal_ray(ray_init):

    qed_root = QEDState(
        rdkit.Chem.MolFromSmiles("C"),
        MoleculeBuilder(max_atoms=5, cache=True),
        smiles="C",
        max_num_actions=20,
        prune_terminal_states=True,
    )

    assert qed_root._using_ray
    assert qed_root.builder._using_ray

    env = GraphEnv({"state": qed_root, "max_num_children": qed_root.max_num_actions})
    assert repr(env.state.children[-1]) == "C (t)"

    # select the terminal state
    obs, reward, terminal, info = env.step(len(env.state.children) - 1)
    assert terminal
    assert np.isclose(reward, 0.3597849378839701)

    obs = env.reset()
    obs, reward, terminal, info = env.step(len(env.state.children) - 1)
    assert not terminal
    assert np.isclose(reward, 0)


def test_observation_space(propane: MoleculeState):
    assert propane.observation_space.contains(propane.observation)


def test_csv_writer(ray_init):

    qed_root = QEDState(
        rdkit.Chem.MolFromSmiles("C"),
        MoleculeBuilder(max_atoms=5, cache=True),
        smiles="CCC",
        max_num_actions=20,
        prune_terminal_states=True,
        force_terminal=True,
        filename="test.csv",
    )
    qed_root.reward

    qed_root.csv_writer.close.remote()

    with open("test.csv", "r") as f:
        assert f.readline().startswith("CCC,")
