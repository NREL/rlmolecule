from typing import Tuple
from pathlib import Path

import numpy as np
import rdkit.Chem
import tensorflow as tf
from alfabet import model

from rlmolecule.molecule.molecule_state import MoleculeState
from examples.gym.molecule_gym.molecule_graph_problem import MoleculeGraphProblem
from examples.gym.molecule_gym.optimize_rad.bde_utils import prepare_for_bde

from examples.stable_radical_optimization.stable_radical_molecule_state import StableRadMoleculeState


model_dir = Path("/projects/rlmolecule/pstjohn/20211020_paper_supplement/models/")
stability_model = Path(model_dir, "stability_model")
redox_model = Path(model_dir, "redox_model")


@tf.function(experimental_relax_shapes=True)
def predict(model: tf.keras.Model, inputs):
    return model.predict_step(inputs)


def windowed_loss(target: float, desired_range: Tuple[float, float]) -> float:
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


class RadGraphProblem(MoleculeGraphProblem):
    def __init__(
        self,
        *args,
        stability_model: Path = stability_model,
        redox_model: Path = redox_model,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.stability_model = tf.keras.models.load_model(stability_model, compile=False)
        self.redox_model = tf.keras.models.load_model(redox_model, compile=False)

    def get_initial_state(self) -> StableRadMoleculeState:
        return StableRadMoleculeState(rdkit.Chem.MolFromSmiles("C"), self.builder)

    def reward(self, state: MoleculeState) -> float:

        model_inputs = {key: tf.constant(np.expand_dims(val, 0)) for key, val in self.make_observation(state).items()}
        spins, buried_vol = predict(self.stability_model, model_inputs)

        spins = spins.numpy().flatten()
        buried_vol = buried_vol.numpy().flatten()

        atom_index = int(spins.argmax())
        max_spin = spins[atom_index]
        spin_buried_vol = buried_vol[atom_index]

        ionization_energy, electron_affinity = predict(self.redox_model, model_inputs).numpy().tolist()[0]

        v_diff = ionization_energy - electron_affinity
        bde, bde_diff = self.calc_bde(state)

        ea_range = (-0.5, 0.2)
        ie_range = (0.5, 1.2)
        v_range = (1, 1.7)
        bde_range = (60, 80)

        stability_score = (1 - max_spin) * 50 + spin_buried_vol
        other_scores = sum(
            (
                windowed_loss(electron_affinity, ea_range),
                windowed_loss(ionization_energy, ie_range),
                windowed_loss(v_diff, v_range),
                windowed_loss(bde, bde_range),
            )
        )

        reward = stability_score + 100 * other_scores / 4

        return reward

    def calc_bde(self, state: MoleculeState):
        """calculate the X-H bde, and the difference to the next-weakest X-H bde in kcal/mol"""

        bde_inputs = prepare_for_bde(state.molecule)
        pred_bdes = model.predict([bde_inputs.mol_smiles], verbose=False, drop_duplicates=False).set_index("bond_index")
        bde_radical = pred_bdes.loc[bde_inputs.bond_index, "bde_pred"]

        if len(bde_inputs.other_h_bonds) == 0:
            bde_diff = 30.0  # Just an arbitrary large number

        else:
            other_h_bdes = pred_bdes.reindex(bde_inputs.other_h_bonds)["bde_pred"]
            bde_diff = (other_h_bdes - bde_radical).min()

        return bde_radical, bde_diff
