import pytest
import ray
import rdkit
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rlmolecule.builder import MoleculeBuilder, count_stereocenters


def to_smiles(input_list):
    return [MolToSmiles(x) for x in input_list]


def test_add_new_atoms_and_bonds():
    from rlmolecule.builder import AddNewAtomsAndBonds

    next_mols = to_smiles(AddNewAtomsAndBonds()([MolFromSmiles("CC=C")]))
    assert len(next_mols) == len(set(next_mols))

    next_mols = list(
        AddNewAtomsAndBonds()([rdkit.Chem.MolFromSmiles("CC(=N)C(=O)C(C)C=N")])
    )
    next_mols_smiles = to_smiles(next_mols)
    assert len(next_mols) == len(set(next_mols_smiles))
    assert len(next_mols) == 42


def test_stereo_enumerator():
    from rlmolecule.builder import StereoEnumerator

    next_mols = list(StereoEnumerator()([MolFromSmiles("CC=CC")]))

    assert len(next_mols) == 2
    for mol in next_mols:
        for bond in mol.GetBonds():
            assert bond.GetStereo() is not rdkit.Chem.rdchem.BondStereo.STEREOANY

    next_mols = to_smiles(StereoEnumerator()([MolFromSmiles("CC(O)(Cl)")]))
    assert "C[C@H](O)Cl" in next_mols
    assert "C[C@@H](O)Cl" in next_mols
    assert len(next_mols) == 2


def test_sa_score():
    from rlmolecule.builder import SAScoreFilter

    next_mols = to_smiles(
        SAScoreFilter(3.5)(
            [MolFromSmiles("CC=CC"), MolFromSmiles("C(Cl)12C3C4C1C5C2C3C45")]
        )
    )
    assert next_mols == ["CC=CC"]


def test_embedding():
    from rlmolecule.builder import EmbeddingFilter

    next_mols = to_smiles(
        EmbeddingFilter()([MolFromSmiles("C1=C=C=1"), MolFromSmiles("CC=CC")])
    )
    assert next_mols


def test_gdb_filter():
    from rlmolecule.builder import GdbFilter

    next_mols = to_smiles(
        GdbFilter()([MolFromSmiles("C1=C=C=1"), MolFromSmiles("CC=CC")])
    )
    assert next_mols == ["CC=CC"]


def test_builder():
    next_mols = to_smiles(MoleculeBuilder()(MolFromSmiles("C=CC")))
    assert next_mols


def test_local_cache():
    builder = MoleculeBuilder(cache=True)
    assert not builder._using_ray
    assert "C=CC" not in builder._builder_cache
    next_mols = to_smiles(builder(MolFromSmiles("C=CC")))
    assert "C=CC" in builder._builder_cache
    next_mols_cache = to_smiles(builder(MolFromSmiles("C=CC")))
    assert set(next_mols) == set(next_mols_cache)


def test_ray_cache(ray_init):
    builder = MoleculeBuilder(cache=True)
    assert builder._using_ray
    assert ray.get(builder._builder_cache.get.remote("C=CC")) is None
    next_mols = to_smiles(builder(MolFromSmiles("C=CC")))
    assert ray.get(builder._builder_cache.get.remote("C=CC")) is not None
    next_mols_cache = to_smiles(builder(MolFromSmiles("C=CC")))
    assert set(next_mols) == set(next_mols_cache)


def test_tautomers():
    from rlmolecule.builder import TautomerCanonicalizer, TautomerEnumerator

    start = rdkit.Chem.MolFromSmiles("CC1=C(O)CCCC1")
    mols = to_smiles(TautomerEnumerator()([start]))
    assert len(mols) == 3
    assert mols[0] != mols[1]

    result = TautomerCanonicalizer()(
        [rdkit.Chem.MolFromSmiles(smiles) for smiles in mols]
    )
    mols_canonical = to_smiles(result)
    assert len(set(mols_canonical)) == 1

    start = rdkit.Chem.MolFromSmiles("CC(=O)C")
    builder_tautomers = MoleculeBuilder(canonicalize_tautomers=True)
    products = to_smiles(builder_tautomers(start))
    assert "CCC(C)=O" in products
    assert "C=C(C)OC" in products


def test_max_atoms():
    builder = MoleculeBuilder(max_atoms=5)
    start = rdkit.Chem.MolFromSmiles("CCCC")
    assert len(builder(start)) >= 1

    builder = MoleculeBuilder(max_atoms=4)
    start = rdkit.Chem.MolFromSmiles("CCCC")
    assert builder(start) == []


def test_parallel_build():
    builder = MoleculeBuilder(
        max_atoms=15,
        min_atoms=4,
        try_embedding=False,
        sa_score_threshold=None,
        stereoisomers=True,
        canonicalize_tautomers=True,
        atom_additions=["C", "N", "O", "S"],
        parallel=True,
    )

    actions = list(builder(rdkit.Chem.MolFromSmiles("CC(=N)C(=O)C(C)C=N")))
    smiles = to_smiles((mol for mol in actions))
    assert len(actions) == len(set(smiles))


# Just make sure this runs in finite time...
@pytest.mark.skip
def test_eagle_error2():
    builder = MoleculeBuilder(
        max_atoms=15,
        min_atoms=4,
        try_embedding=True,
        sa_score_threshold=None,
        stereoisomers=True,
        canonicalize_tautomers=True,
        atom_additions=["C", "N", "O", "S"],
        parallel=True,
    )

    mol = rdkit.Chem.MolFromSmiles("Cc1nc(-c2cc(=O)[nH][nH]2)c[nH]1")
    list(builder(mol))


def test_eagle_error3():
    """This one makes sure that stereoisomers and tautomerization work together"""

    builder = MoleculeBuilder(
        max_atoms=15,
        min_atoms=4,
        try_embedding=True,
        sa_score_threshold=None,
        stereoisomers=True,
        canonicalize_tautomers=True,
        atom_additions=["C", "N", "O", "S"],
    )

    mol = rdkit.Chem.MolFromSmiles("CC(N)S")
    actions = list(builder(mol))
    smiles = to_smiles((state for state in actions))
    for smiles in smiles:
        stereo_count = count_stereocenters(smiles)
        assert stereo_count["atom_unassigned"] == 0, f"{smiles}"
        assert stereo_count["bond_unassigned"] == 0, f"{smiles}"
