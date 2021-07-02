from tempfile import TemporaryDirectory

import rdkit
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles

from rlmolecule.molecule.builder.builder import MoleculeBuilder
from rlmolecule.molecule.molecule_state import MoleculeState


def to_smiles(input_list):
    return [MolToSmiles(x) for x in input_list]


def test_add_new_atoms_and_bonds():
    from rlmolecule.molecule.builder.builder import AddNewAtomsAndBonds
    next_mols = to_smiles(AddNewAtomsAndBonds()([MolFromSmiles('CC=C')]))
    assert len(next_mols) == len(set(next_mols))

    next_mols = list(AddNewAtomsAndBonds()([rdkit.Chem.MolFromSmiles('CC(=N)C(=O)C(C)C=N')]))
    next_mols_smiles = to_smiles(next_mols)
    assert len(next_mols) == len(set(next_mols_smiles))
    assert len(next_mols) == 42


def test_stereo_enumerator():
    from rlmolecule.molecule.builder.builder import StereoEnumerator
    next_mols = list(StereoEnumerator()([MolFromSmiles('CC=CC')]))

    assert len(next_mols) == 2
    for mol in next_mols:
        for bond in mol.GetBonds():
            assert bond.GetStereo() is not rdkit.Chem.rdchem.BondStereo.STEREOANY

    next_mols = to_smiles(StereoEnumerator()([MolFromSmiles('CC(O)(Cl)')]))
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


def test_cache():
    from rlmolecule.molecule.builder.builder import MoleculeBuilder
    with TemporaryDirectory() as tempdir:
        next_mols = to_smiles(MoleculeBuilder(cache_dir=tempdir)(MolFromSmiles('C=CC')))
        next_mols = to_smiles(MoleculeBuilder(cache_dir=tempdir, num_shards=2)(MolFromSmiles('C=CC')))

    assert next_mols


def test_tautomers():
    from rlmolecule.molecule.builder.builder import MoleculeBuilder, TautomerCanonicalizer, TautomerEnumerator

    start = rdkit.Chem.MolFromSmiles('CC1=C(O)CCCC1')
    mols = to_smiles(TautomerEnumerator()([start]))
    assert len(mols) == 3
    assert mols[0] != mols[1]

    result = TautomerCanonicalizer()([rdkit.Chem.MolFromSmiles(smiles) for smiles in mols])
    mols_canonical = to_smiles(result)
    assert len(set(mols_canonical)) == 1

    builder_tautomers = MoleculeBuilder(canonicalize_tautomers=True)
    products = to_smiles(builder_tautomers(start))


def test_parallel_build():
    builder = MoleculeBuilder(max_atoms=15,
                              min_atoms=4,
                              try_embedding=False,
                              sa_score_threshold=None,
                              stereoisomers=True,
                              canonicalize_tautomers=True,
                              atom_additions=['C', 'N', 'O', 'S'],
                              parallel=True
                              )

    state = MoleculeState(rdkit.Chem.MolFromSmiles('CC(=N)C(=O)C(C)C=N'), builder=builder)
    actions = state.get_next_actions()
    smiles = to_smiles((state.molecule for state in actions))
    assert len(actions) == len(set(smiles))


# Just make sure this runs in finite time...
def test_eagle_error2():

    builder = MoleculeBuilder(max_atoms=15,
                              min_atoms=4,
                              try_embedding=True,
                              sa_score_threshold=None,
                              stereoisomers=True,
                              canonicalize_tautomers=True,
                              atom_additions=['C', 'N', 'O', 'S'],
                              parallel=True
                              )

    mol = rdkit.Chem.MolFromSmiles('Cc1nc(-c2cc(=O)[nH][nH]2)c[nH]1')
    state = MoleculeState(mol, builder=builder)
    actions = state.get_next_actions()