import pytest
from rdkit.Chem.rdmolfiles import MolFromSmiles

from rlmolecule.molecule.builder.builder import MoleculeBuilder


@pytest.fixture
def molecule():
    return MolFromSmiles('CCC')


@pytest.fixture
def builder():
    return MoleculeBuilder()


def test_equals(molecule, builder):
    from rlmolecule.molecule.molecule_state import MoleculeState
    assert MoleculeState(molecule, builder) == MoleculeState(MolFromSmiles('CCC'), builder)


def test_num_atoms(molecule, builder):
    from rlmolecule.molecule.molecule_state import MoleculeState
    assert MoleculeState(molecule, builder).num_atoms == 3


def test_get_next_actions(molecule):
    from rlmolecule.molecule.molecule_state import MoleculeState

    builder = MoleculeBuilder(max_atoms=5)
    next_actions = set(MoleculeState(molecule, builder).get_next_actions())
    assert MoleculeState(MolFromSmiles('CC(C)=N'), builder) in next_actions
    assert MoleculeState(MolFromSmiles('CCC=O'), builder) in next_actions

    builder = MoleculeBuilder(max_atoms=3)
    next_actions = set(MoleculeState(molecule, builder).get_next_actions())
    assert len(next_actions) == 0


def test_serialize_deserialize(molecule):
    from rlmolecule.molecule.molecule_state import MoleculeState

    builder = MoleculeBuilder(max_atoms=5)
    mol_state = MoleculeState(molecule, builder)
    next_actions = list(mol_state.get_next_actions())
    string_molecule = mol_state.serialize()
    new_mol = MoleculeState.deserialize(string_molecule)

    assert new_mol == mol_state
    assert list(new_mol.get_next_actions()) == next_actions
