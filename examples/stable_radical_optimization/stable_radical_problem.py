import os
import pathlib
from typing import Tuple

import numpy as np
import rdkit
import tensorflow as tf
from rdkit import Chem

from bde_utils import bde_get_inputs, prepare_for_bde
from examples.stable_radical_optimization.stable_radical_molecule_state import StableRadMoleculeState, \
    MoleculeBuilderProtectRadical
from rlmolecule.molecule.builder.builder import MoleculeBuilder
from rlmolecule.molecule.molecule_problem import MoleculeTFAlphaZeroProblem
from rlmolecule.molecule.molecule_state import MoleculeState
from rlmolecule.sql.run_config import RunConfig
from rlmolecule.tree_search.metrics import collect_metrics
from rlmolecule.tree_search.reward import RankedRewardFactory


@tf.function(experimental_relax_shapes=True)
def predict(model: 'tf.keras.Model', inputs):
    return model.predict_step(inputs)


def windowed_loss(target: float, desired_range: Tuple[float, float]) -> float:
    """ Returns 0 if the molecule is in the middle of the desired range,
    scaled loss otherwise. """

    span = desired_range[1] - desired_range[0]

    lower_lim = desired_range[0] + span / 6
    upper_lim = desired_range[1] - span / 6

    if target < lower_lim:
        return max(1 - 3 * (abs(target - lower_lim) / span), 0)
    elif target > upper_lim:
        return max(1 - 3 * (abs(target - upper_lim) / span), 0)
    else:
        return 1


class StableRadOptProblem(MoleculeTFAlphaZeroProblem):
    def __init__(self,
                 engine: 'sqlalchemy.engine.Engine',
                 builder: 'MoleculeBuilder',
                 stability_model: 'tf.keras.Model',
                 redox_model: 'tf.keras.Model',
                 bde_model: 'tf.keras.Model',
                 initial_state: str,
                 **kwargs) -> None:
        """A class to estimate the suitability of radical species in redox flow batteries.

        :param engine: A sqlalchemy engine pointing to a suitable database backend
        :param builder: A MoleculeBuilder class to handle molecule construction
        :param stability_model: A tensorflow model to estimate spin and buried volumes
        :param redox_model: A tensorflow model to estimate electron affinity and ionization energies
        :param bde_model: A tensorflow model to estimate bond dissociation energies
        :param initial_state: The initial starting state for the molecule search.
        """
        self.initial_state = initial_state
        self.engine = engine
        self._builder = builder
        self.stability_model = stability_model
        self.redox_model = redox_model
        self.bde_model = bde_model
        super(StableRadOptProblem, self).__init__(engine, builder, **kwargs)

    def get_initial_state(self) -> MoleculeState:
        if self.initial_state == 'C':
            return StableRadMoleculeState(rdkit.Chem.MolFromSmiles('C'), self._builder)
        else:
            return MoleculeState(rdkit.Chem.MolFromSmiles(self.initial_state), self._builder)

    def get_reward(self, state: MoleculeState) -> Tuple[float, dict]:
        policy_inputs = self.get_policy_inputs(state)

        # Node is outside the domain of validity
        if (policy_inputs['atom'] == 1).any() | (policy_inputs['bond'] == 1).any():
            return 0.0, {'forced_terminal': False, 'smiles': state.smiles}

        if state.forced_terminal:
            reward, stats = self.calc_reward(state)
            stats.update({'forced_terminal': True, 'smiles': state.smiles})
            return reward, stats

        # Reward called on a non-terminal state, likely built into a corner
        return 0.0, {'forced_terminal': False, 'smiles': state.smiles}

    @collect_metrics
    def calc_reward(self, state: MoleculeState) -> (float, {}):
        """
        """
        model_inputs = {key: tf.constant(np.expand_dims(val, 0)) for key, val in self.get_policy_inputs(state).items()}
        spins, buried_vol = predict(self.stability_model, model_inputs)

        spins = spins.numpy().flatten()
        buried_vol = buried_vol.numpy().flatten()

        atom_index = int(spins.argmax())
        max_spin = spins[atom_index]
        spin_buried_vol = buried_vol[atom_index]

        atom_type = state.molecule.GetAtomWithIdx(atom_index).GetSymbol()

        ionization_energy, electron_affinity = predict(self.redox_model, model_inputs).numpy().tolist()[0]

        v_diff = ionization_energy - electron_affinity
        bde, bde_diff = self.calc_bde(state)

        ea_range = (-.5, 0.2)
        ie_range = (.5, 1.2)
        v_range = (1, 1.7)
        bde_range = (60, 80)

        reward = ((1 - max_spin) * 50 + spin_buried_vol + 100 *
                  (windowed_loss(electron_affinity, ea_range) + windowed_loss(ionization_energy, ie_range) +
                   windowed_loss(v_diff, v_range) + windowed_loss(bde, bde_range)) / 4)

        stats = {
            'max_spin': max_spin,
            'spin_buried_vol': spin_buried_vol,
            'ionization_energy': ionization_energy,
            'electron_affinity': electron_affinity,
            'bde': bde,
            'bde_diff': bde_diff,
        }
        stats = {key: str(val) for key, val in stats.items()}

        return reward, stats

    def calc_bde(self, state: MoleculeState):
        """calculate the X-H bde, and the difference to the next-weakest X-H bde in kcal/mol"""

        bde_inputs = prepare_for_bde(state.molecule)
        # model_inputs = self.bde_get_inputs(state.molecule)
        model_inputs = bde_get_inputs(bde_inputs.mol_smiles)

        pred_bdes = predict(self.bde_model, model_inputs)
        pred_bdes = pred_bdes[0][0, :, 0].numpy()

        bde_radical = pred_bdes[bde_inputs.bond_index]

        if len(bde_inputs.other_h_bonds) == 0:
            bde_diff = 30.  # Just an arbitrary large number

        else:
            other_h_bdes = pred_bdes[bde_inputs.other_h_bonds]
            bde_diff = (other_h_bdes - bde_radical).min()

        return bde_radical, bde_diff


def construct_problem(run_config: RunConfig, stability_model: pathlib.Path, redox_model: pathlib.Path,
                      bde_model: pathlib.Path, **kwargs):
    stability_model = tf.keras.models.load_model(stability_model, compile=False)
    redox_model = tf.keras.models.load_model(redox_model, compile=False)
    bde_model = tf.keras.models.load_model(bde_model, compile=False)

    prob_config = run_config.problem_config
    initial_state = prob_config.get('initial_state', 'C')
    if initial_state == 'C':
        builder_class = MoleculeBuilder
    else:
        builder_class = MoleculeBuilderProtectRadical

    builder = builder_class(max_atoms=prob_config.get('max_atoms', 15),
                            min_atoms=prob_config.get('min_atoms', 4),
                            tryEmbedding=prob_config.get('tryEmbedding', True),
                            sa_score_threshold=prob_config.get('sa_score_threshold', 3.5),
                            stereoisomers=prob_config.get('stereoisomers', True),
                            atom_additions=prob_config.get('atom_additions', ('C', 'N', 'O', 'S')))

    engine = run_config.start_engine()

    run_id = run_config.run_id
    train_config = run_config.train_config
    reward_factory = RankedRewardFactory(engine=engine,
                                         run_id=run_id,
                                         reward_buffer_min_size=train_config.get('reward_buffer_min_size', 50),
                                         reward_buffer_max_size=train_config.get('reward_buffer_max_size', 250),
                                         ranked_reward_alpha=train_config.get('ranked_reward_alpha', 0.75))

    problem = StableRadOptProblem(
        engine,
        builder,
        stability_model,
        redox_model,
        bde_model,
        run_id=run_id,
        initial_state=initial_state,
        reward_class=reward_factory,
        features=train_config.get('features', 64),
        # Number of attention heads
        num_heads=train_config.get('num_heads', 4),
        num_messages=train_config.get('num_messages', 3),
        max_buffer_size=train_config.get('max_buffer_size', 200),
        # Don't start training the model until this many games have occurred
        min_buffer_size=train_config.get('min_buffer_size', 15),
        batch_size=train_config.get('batch_size', 32),
        policy_checkpoint_dir=os.path.join(train_config.get('policy_checkpoint_dir', 'policy_checkpoints'), run_id))

    return problem
