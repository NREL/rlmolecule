import tempfile

import pytest
import ray
import rdkit
from ray.rllib.utils.framework import try_import_tf
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.qed import QEDState
from rlmolecule.molecule_state import MoleculeData

tf1, tf, tfv = try_import_tf(error=True)
assert tfv == 2
if not tf1.executing_eagerly():
    tf1.enable_eager_execution()


@pytest.fixture
def ray_init():
    ray.init(num_cpus=1, local_mode=True)
    yield None
    ray.shutdown()


@pytest.fixture
def builder() -> MoleculeBuilder:
    return MoleculeBuilder(max_atoms=5)


@pytest.fixture(scope="class")
def tmpdirname():
    """
    A directory for the checkpoint files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def qed_root(builder: MoleculeBuilder) -> QEDState:
    return QEDState(
        rdkit.Chem.MolFromSmiles("C"),
        data=MoleculeData(builder, max_num_actions=20),
        smiles="C",
    )
