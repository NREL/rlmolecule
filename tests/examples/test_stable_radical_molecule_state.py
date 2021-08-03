import rdkit.Chem


def test_get_fingerprint():
    from examples.stable_radical_optimization.stable_radical_molecule_state import FingerprintFilter
    filter = FingerprintFilter()
    mol = rdkit.Chem.MolFromSmiles('C/C(=C(C(/O)=C/[O])\C(C)(C)C)C(C)(C)C')
    fps = set(filter.get_fingerprint(mol))


def test_filter():
    from examples.stable_radical_optimization.stable_radical_molecule_state import FingerprintFilter
    filter = FingerprintFilter()
    mol = rdkit.Chem.MolFromSmiles('C/C(=C(C(/O)=C/[O])\C(C)(C)C)C(C)(C)C')
    assert not filter.filter(mol)

    mol = rdkit.Chem.MolFromSmiles('C/C(=C(C(/O)=C/[O])\C(C)(C)C)C(C)C')
    assert filter.filter(mol)
