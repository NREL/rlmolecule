import pytest
from rdkit.Chem.rdmolfiles import MolFromSmiles


@pytest.fixture
def molecule():
    return MolFromSmiles('CCC')


@pytest.fixture
def config():
    from rlmolecule.molecule.molecule_config import MoleculeConfig
    return MoleculeConfig()


def test_equals(molecule, config):
    from rlmolecule.molecule.molecule_state import MoleculeState
    assert MoleculeState(molecule, config) == MoleculeState(MolFromSmiles('CCC'), config)


def test_num_atoms(molecule, config):
    from rlmolecule.molecule.molecule_state import MoleculeState
    assert MoleculeState(molecule, config).num_atoms == 3


def test_get_next_actions(molecule):
    from rlmolecule.molecule.molecule_state import MoleculeState
    from rlmolecule.molecule.molecule_config import MoleculeConfig

    config = MoleculeConfig(max_atoms=5)
    next_actions = set(MoleculeState(molecule, config).get_next_actions())
    assert MoleculeState(MolFromSmiles('CC(C)=N'), config) in next_actions
    assert MoleculeState(MolFromSmiles('CCC=O'), config) in next_actions

    config = MoleculeConfig(max_atoms=3)
    next_actions = set(MoleculeState(molecule, config).get_next_actions())
    assert len(next_actions) == 0


def test_serialize_deserialize(molecule):
    from rlmolecule.molecule.molecule_state import MoleculeState
    from rlmolecule.molecule.molecule_config import MoleculeConfig

    config = MoleculeConfig(max_atoms=5)
    mol_state = MoleculeState(molecule, config)
    next_actions = list(mol_state.get_next_actions())
    string_molecule = mol_state.serialize()
    assert type(string_molecule) == str
    new_mol = MoleculeState.deserialize(string_molecule)
    assert list(new_mol.get_next_actions()) == next_actions
