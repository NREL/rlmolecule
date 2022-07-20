import logging
from typing import Optional
import rdkit
from rdkit import Chem
from typing import Tuple
from typing import Dict, Optional

import tensorflow as tf
import ray
import sys
from rlmolecule.actors import CSVActorWriter
from rlmolecule.molecule_state import MoleculeState
from nfp.preprocessing import MolPreprocessor
from rlmolecule.policy.preprocessor import load_preprocessor

logger = logging.getLogger(__name__)

sys.path.append("/projects/rlmolecule/pstjohn/models/20201031_bde/")
from preprocess_inputs import preprocessor as bde_preprocessor

bde_preprocessor.from_json(
    "/projects/rlmolecule/pstjohn/models/20201031_bde/preprocessor.json"
)

def get_csv_logger(filename):
    return CSVActorWriter.options(
        name="redox_results", lifetime="detached", get_if_exists=True
    ).remote(filename)


class RedoxState(MoleculeState):
    def __init__(
        self,
        *args,
        use_ray: Optional[bool] = None,
        filename: str = "redox_results.csv",
        stability_model: tf.keras.Model,
        redox_model: tf.keras.Model,
        bde_model: tf.keras.Model,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if use_ray is None:
            use_ray = ray.is_initialized()

        self._using_ray = use_ray
        self.filename = filename

        self.stability_model = tf.keras.models.load_model(
            stability_model, compile=False
        )
        self.redox_model = tf.keras.models.load_model(redox_model, compile=False)
        self.bde_model = tf.keras.models.load_model(bde_model, compile=False)

        self.preprocessor = load_preprocessor()

        if self._using_ray:
            self.csv_writer = get_csv_logger(filename)
        else:
            self.csv_writer = None
            
        print(self.stability_model, self.redox_model, self.bde_model)

    def new(self, *args, **kwargs):
        return super().new(
            *args, **kwargs, use_ray=self._using_ray, filename=self.filename
        )

    @property
    def reward(self) -> float:
        if self.forced_terminal:
            reward, stats = self.calc_reward(self.molecule)
            if self.csv_writer is not None:
                self.csv_writer.write.remote([self.smiles, reward, stats])
            else:
                logger.info(f"Redox: {self.smiles} - {reward} - {stats}")
            return reward
        else:
            return 0.0

    def calc_reward(self) -> float:
        """ """
        model_inputs = {
            key: tf.constant(np.expand_dims(val, 0))
            for key, val in self.get_policy_inputs(self.molecule).items()
        }
        spins, buried_vol = predict(self.stability_model, model_inputs)

        spins = spins.numpy().flatten()
        buried_vol = buried_vol.numpy().flatten()

        atom_index = int(spins.argmax())
        max_spin = spins[atom_index]
        spin_buried_vol = buried_vol[atom_index]

        atom_type = self.molecule.GetAtomWithIdx(atom_index).GetSymbol()

        ionization_energy, electron_affinity = (
            predict(self.redox_model, model_inputs).numpy().tolist()[0]
        )

        v_diff = ionization_energy - electron_affinity
        bde, bde_diff = self.calc_bde(self.molecule)

        ea_range = (-0.5, 0.2)
        ie_range = (0.5, 1.2)
        v_range = (1, 1.7)
        bde_range = (60, 80)

        # This is a bit of a placeholder; but the range for spin is about 1/50th that
        # of buried volume.
        reward = (
            (1 - max_spin) * 50
            + spin_buried_vol
            + 100
            * (
                self.windowed_loss(electron_affinity, ea_range)
                + self.windowed_loss(ionization_energy, ie_range)
                + self.windowed_loss(v_diff, v_range)
                + self.windowed_loss(bde, bde_range)
            )
            / 4
        )
        # the addition of bde_diff was to help ensure that
        # the stable radical had the lowest bde in the molecule
        # + 25 / (1 + np.exp(-(bde_diff - 10)))

        stats = {
            "max_spin": max_spin,
            "spin_buried_vol": spin_buried_vol,
            "ionization_energy": ionization_energy,
            "electron_affinity": electron_affinity,
            "bde": bde,
            "bde_diff": bde_diff,
        }

        stats = {key: str(val) for key, val in stats.items()}

        return reward, stats

    def calc_bde(self):
        """calculate the X-H bde, and the difference to the next-weakest X-H bde in kcal/mol"""

        bde_inputs = self.prepare_for_bde(self.molecule)
        # model_inputs = self.bde_get_inputs(state.molecule)
        model_inputs = self.bde_get_inputs(bde_inputs.mol_smiles)

        pred_bdes = predict(self.bde_model, model_inputs)
        pred_bdes = pred_bdes[0][0, :, 0].numpy()

        bde_radical = pred_bdes[bde_inputs.bond_index]

        if len(bde_inputs.other_h_bonds) == 0:
            bde_diff = 30.0  # Just an arbitrary large number

        else:
            other_h_bdes = pred_bdes[bde_inputs.other_h_bonds]
            bde_diff = (other_h_bdes - bde_radical).min()

        return bde_radical, bde_diff

    def prepare_for_bde(self, mol: rdkit.Chem.Mol):

        radical_index = None
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetNumRadicalElectrons() != 0:
                assert radical_index == None
                is_radical = True
                radical_index = i

                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                atom.SetNumRadicalElectrons(0)
                break

        radical_rank = Chem.CanonicalRankAtoms(mol, includeChirality=True)[
            radical_index
        ]

        mol_smiles = Chem.MolToSmiles(mol)
        # TODO this line seems redundant
        mol = Chem.MolFromSmiles(mol_smiles)

        radical_index_reordered = list(
            Chem.CanonicalRankAtoms(mol, includeChirality=True)
        ).index(radical_rank)

        molH = Chem.AddHs(mol)
        for bond in molH.GetAtomWithIdx(radical_index_reordered).GetBonds():
            if "H" in {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}:
                bond_index = bond.GetIdx()
                break

        h_bond_indices = [
            bond.GetIdx()
            for bond in filter(
                lambda bond: (
                    (bond.GetEndAtom().GetSymbol() == "H")
                    | (bond.GetBeginAtom().GetSymbol() == "H")
                ),
                molH.GetBonds(),
            )
        ]

        other_h_bonds = list(set(h_bond_indices) - {bond_index})

        return pd.Series(
            {
                "mol_smiles": mol_smiles,
                "radical_index_mol": radical_index_reordered,
                "bond_index": bond_index,
                "other_h_bonds": other_h_bonds,
            }
        )

    def bde_get_inputs(self, mol_smiles):
        """The BDE model was trained on a different set of data
        so we need to use corresponding preprocessor here
        """
        inputs = bde_preprocessor.construct_feature_matrices(mol_smiles, train=False)
        assert not (inputs["atom"] == 1).any() | (inputs["bond"] == 1).any()
        return {key: tf.constant(np.expand_dims(val, 0)) for key, val in inputs.items()}

    def windowed_loss(self, target: float, desired_range: Tuple[float, float]) -> float:
        """Returns 0 if the molecule is in the middle of the desired range,
        scaled loss otherwise."""

        span = desired_range[1] - desired_range[0]

        lower_lim = desired_range[0] + span / 6
        upper_lim = desired_range[1] - span / 6

        if target < lower_lim:
            return max(1 - 3 * (abs(target - lower_lim) / span), 0)
        elif target > upper_lim:
            return max(1 - 3 * (abs(target - upper_lim) / span), 0)
        else:
            return 1

    @tf.function(experimental_relax_shapes=True)
    def predict(model: "tf.keras.Model", inputs):
        return model.predict_step(inputs)

    def get_policy_inputs(self) -> Dict:
        return self.preprocessor(self.molecule)

    def __setstate__(self, d):
        super().__setstate__(d)
        if d["_using_ray"]:
            self.csv_writer = get_csv_logger(d["filename"])

    def __getstate__(self):
        data = super().__getstate__()
        data["csv_writer"] = None
        return data
