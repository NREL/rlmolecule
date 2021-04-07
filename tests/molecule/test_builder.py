import rdkit
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles


def to_smiles(input_list):
    return [MolToSmiles(x) for x in input_list]


def test_add_new_atoms_and_bonds():
    from rlmolecule.molecule.builder.builder import AddNewAtomsAndBonds
    next_mols = to_smiles(AddNewAtomsAndBonds()([MolFromSmiles('CC=C')]))
    assert len(next_mols) == len(set(next_mols))


def test_stereo_enumerator():
    from rlmolecule.molecule.builder.builder import StereoEnumerator
    next_mols = list(StereoEnumerator()([MolFromSmiles('CC=CC')]))

    assert len(next_mols) == 2
    for mol in next_mols:
        for bond in mol.GetBonds():
            assert bond.GetStereo() is not rdkit.Chem.rdchem.BondStereo.STEREOANY

    next_mols = to_smiles(StereoEnumerator()([MolFromSmiles('C(C)(O)(Cl)')]))
    assert 'C[C@H](O)Cl' in next_mols
    assert 'C[C@@H](O)Cl' in next_mols
    assert len(next_mols) == 2


def test_sa_score():
    from rlmolecule.molecule.builder.builder import SAScoreFilter
    next_mols = to_smiles(SAScoreFilter(3.5)([MolFromSmiles('CC=CC'), MolFromSmiles('C(Cl)12C3C4C1C5C2C3C45')]))
    assert next_mols == ['CC=CC']


def test_embedding():
    from rlmolecule.molecule.builder.builder import EmbeddingFilter
    next_mols = to_smiles(EmbeddingFilter()([MolFromSmiles('C1=C=C=1'), MolFromSmiles('CC=CC')]))
    assert next_mols


def test_gdb_filter():
    from rlmolecule.molecule.builder.builder import GdbFilter
    next_mols = to_smiles(GdbFilter()([MolFromSmiles('C1=C=C=1'), MolFromSmiles('CC=CC')]))
    assert next_mols == ['CC=CC']


def test_builder():
    from rlmolecule.molecule.builder.builder import MoleculeBuilder
    next_mols = to_smiles(MoleculeBuilder()(MolFromSmiles('C=CC')))
    assert next_mols