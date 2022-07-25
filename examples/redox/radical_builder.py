import gzip
import os
import pickle

import rdkit
from rdkit.Chem import FragmentCatalog
from rlmolecule.builder import AddNewAtomsAndBonds, MoleculeBuilder, MoleculeFilter

fcgen = FragmentCatalog.FragCatGenerator()
fpgen = FragmentCatalog.FragFPGenerator()
dir_path = os.path.dirname(os.path.realpath(__file__))


def build_radicals(starting_mol):
    """Build organic radicals."""

    generated_smiles = set()

    for i, atom in enumerate(starting_mol.GetAtoms()):
        if AddNewAtomsAndBonds._get_free_valence(atom) > 0:
            rw_mol = rdkit.Chem.RWMol(starting_mol)
            rw_mol.GetAtomWithIdx(i).SetNumRadicalElectrons(1)

            rdkit.Chem.SanitizeMol(rw_mol)
            smiles = rdkit.Chem.MolToSmiles(rw_mol)
            if smiles not in generated_smiles:
                # This makes sure the atom ordering is standardized
                yield rdkit.Chem.MolFromSmiles(smiles)
                generated_smiles.add(smiles)


class FingerprintFilter(MoleculeFilter):
    def __init__(self):
        super(FingerprintFilter, self).__init__()
        with gzip.open(os.path.join(dir_path, "redox_fragment_data.pz")) as f:
            data = pickle.load(f)
        self.fcat = data["fcat"]
        self.valid_fps = set(data["valid_fps"])

    def get_fingerprint(self, mol):
        fcgen.AddFragsFromMol(mol, self.fcat)
        fp = fpgen.GetFPForMol(mol, self.fcat)
        for i in fp.GetOnBits():
            yield self.fcat.GetEntryDescription(i)

    def filter(self, molecule: rdkit.Chem.Mol) -> bool:
        fps = set(self.get_fingerprint(molecule))
        if fps.difference(self.valid_fps) == set():
            return True
        else:
            return False


class MoleculeBuilderWithFingerprint(MoleculeBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation_stack += [FingerprintFilter()]
