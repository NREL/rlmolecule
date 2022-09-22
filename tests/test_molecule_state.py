import csv
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import rdkit
from graphenv.graph_env import GraphEnv
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.qed import QEDState
from rlmolecule.molecule_state import MoleculeData, MoleculeState


@pytest.fixture
def propane(builder: MoleculeBuilder) -> MoleculeState:
    return MoleculeState(
        rdkit.Chem.MolFromSmiles("CCC"),
        MoleculeData(builder),
        force_terminal=False,
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
        MoleculeData(builder, max_num_actions=20, prune_terminal_states=True),
        smiles="C",
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
        MoleculeData(
            MoleculeBuilder(max_atoms=5, cache=True),
            max_num_actions=20,
            prune_terminal_states=True,
        ),
        smiles="C",
    )

    assert qed_root.data.using_ray
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


def test_csv_writer(ray_init, caplog):
    caplog.set_level(logging.INFO)

    with TemporaryDirectory() as tempdir:

        data = MoleculeData(
            MoleculeBuilder(max_atoms=5, cache=True),
            max_num_actions=20,
            prune_terminal_states=True,
            log_reward_filepath=Path(tempdir, "test.csv"),
        )

        state = QEDState(rdkit.Chem.MolFromSmiles("CCC"), data, force_terminal=True)
        state.reward

        state = QEDState(rdkit.Chem.MolFromSmiles("CCO"), data, force_terminal=True)
        state.reward

        state.data.csv_writer.close.remote()

        with open(Path(tempdir, "test.csv")) as f:
            csvdata = list(csv.reader(f))

        assert csvdata[0][0] == "CCC"
        assert np.isclose(float(csvdata[0][1]), 0.3854706587740357)

        assert csvdata[1][0] == "CCO"
        assert np.isclose(float(csvdata[1][1]), 0.40680796565539457)
