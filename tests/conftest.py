import tempfile

import pytest
import ray
import rdkit
from ray.rllib.utils.framework import try_import_tf
from rdkit.Chem.QED import qed
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.molecule_state import MoleculeState

tf1, tf, tfv = try_import_tf(error=True)
assert tfv == 2
if not tf1.executing_eagerly():
    tf1.enable_eager_execution()


@pytest.fixture(scope="module")
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


class QEDState(MoleculeState):
    @property
    def reward(self) -> float:
        if self.forced_terminal:
            return qed(self.molecule)
        else:
            return 0.0


@pytest.fixture
def qed_root(builder: MoleculeBuilder) -> QEDState:
    return QEDState(rdkit.Chem.MolFromSmiles("C"), builder)
