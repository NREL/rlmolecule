import pytest
from rdkit.Chem.rdmolfiles import MolFromSmiles


@pytest.fixture
def molecule():
    return MolFromSmiles('CCC')


def test_equals(molecule):
    from rlmolecule.molecule.molecule_state import MoleculeState
    assert MoleculeState(molecule) == MoleculeState(MolFromSmiles('CCC'))


def test_num_atoms(molecule):
    from rlmolecule.molecule.molecule_state import MoleculeState
    assert MoleculeState(molecule).num_atoms == 3


def test_get_next_actions(molecule):
    from rlmolecule.molecule.molecule_state import MoleculeState, MoleculeConfig

    config = MoleculeConfig(max_atoms=5)
    next_actions = set(MoleculeState(molecule, config).get_next_actions())
    assert not MoleculeState(molecule).terminal
    assert MoleculeState(MolFromSmiles('CC(C)=N')) in next_actions
    assert MoleculeState(MolFromSmiles('CCC=O')) in next_actions

    config = MoleculeConfig(max_atoms=3)
    assert MoleculeState(molecule, config).terminal == True

