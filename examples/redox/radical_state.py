from typing import Sequence, Tuple

import rdkit
from examples.redox.models import expand_inputs
from examples.redox.models import preprocessor as redox_preprocessor
from examples.redox.models import redox_model, stability_model
from examples.redox.radical_builder import build_radicals
from graphenv.vertex import V
from rdkit import Chem
from rlmolecule.molecule_state import MoleculeState

from alfabet.prediction import model as bde_model  # isort:skip
from alfabet.preprocessor import preprocessor as bde_preprocessor  # isort:skip


class RadicalState(MoleculeState):
    def _get_terminal_actions(self) -> Sequence[V]:
        """For the radical optimization, each 'terminal' state chooses an atom to be a
        radical center

        Returns:
            Sequence[V]: A list of terminal states
        """
        return [
            self.new(radical, force_terminal=True)
            for radical in build_radicals(self.molecule)
        ]

    @property
    def reward(self) -> float:
        if self.forced_terminal:
            reward, stats = self.calc_reward(self.molecule)
            self.data.log_reward([self.smiles, reward, stats])
            return reward
        else:
            return 0.0

    def calc_reward(self) -> float:
        """ """

        model_inputs = expand_inputs(redox_preprocessor(self.molecule))

        spins, buried_vol = stability_model.predict_step(model_inputs)
        ionization_energy, electron_affinity = (
            redox_model.predict_step(model_inputs).numpy().tolist()[0]
        )

        spins = spins.numpy().flatten()
        buried_vol = buried_vol.numpy().flatten()

        atom_index = int(spins.argmax())
        max_spin = spins[atom_index]
        spin_buried_vol = buried_vol[atom_index]

        # atom_type = self.molecule.GetAtomWithIdx(atom_index).GetSymbol()

        v_diff = ionization_energy - electron_affinity
        bde, bde_diff = self.calc_bde()

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
        """calculate the X-H bde, and the difference to the next-weakest X-H bde in
        kcal/mol"""

        bde_inputs = self.prepare_for_bde()
        model_inputs = expand_inputs(bde_preprocessor(bde_inputs["mol_smiles"]))

        pred_bdes = bde_model.predict(model_inputs, verbose=0)
        pred_bdes = pred_bdes[0][0, :, 0]

        bde_radical = pred_bdes[bde_inputs["bond_index"]]

        if len(bde_inputs["other_h_bonds"]) == 0:
            bde_diff = 30.0  # Just an arbitrary large number

        else:
            other_h_bdes = pred_bdes[bde_inputs["other_h_bonds"]]
            bde_diff = (other_h_bdes - bde_radical).min()

        return bde_radical, bde_diff

    def prepare_for_bde(self):

        mol = rdkit.Chem.Mol(self.molecule)
        radical_index = None
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetNumRadicalElectrons() != 0:
                assert radical_index is None
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

        return {
            "mol_smiles": mol_smiles,
            "radical_index_mol": radical_index_reordered,
            "bond_index": bond_index,
            "other_h_bonds": other_h_bonds,
        }

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
