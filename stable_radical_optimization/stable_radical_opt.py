import argparse
import pathlib
import logging
import time
from typing import Tuple

import sqlalchemy
from sqlalchemy import create_engine

import numpy as np
import pandas as pd
import nfp

import rdkit
from rdkit import Chem
from rdkit.Chem import Mol, MolToSmiles

from rlmolecule.tree_search.reward import LinearBoundedRewardFactory
from rlmolecule.molecule.policy import preprocessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#sys.path.append('/projects/rlmolecule/pstjohn/models/20201031_bde/')
#from preprocess_inputs import preprocessor as bde_preprocessor
#bde_preprocessor.from_json('/projects/rlmolecule/pstjohn/models/20201031_bde/preprocessor.json')
# TODO would we need a different preprocessor.json file here
# than we have in rlmolecule.molecule.policy.preprocessor.json?
bde_preprocessor = preprocessor.load_preprocessor()


def construct_problem(
        stability_model: pathlib.Path,
        redox_model: pathlib.Path,
        bde_model: pathlib.Path,
        **kwargs):
    # We have to delay all importing of tensorflow until the child processes launch,
    # see https://github.com/tensorflow/tensorflow/issues/8220. We should be more careful about where / when we
    # import tensorflow, especially if there's a chance we'll use tf.serving to do the policy / reward evaluations on
    # the workers. Might require upstream changes to nfp as well.
    from rlmolecule.tree_search.reward import RankedRewardFactory
    from rlmolecule.molecule.molecule_config import MoleculeConfig
    from rlmolecule.molecule.molecule_problem import MoleculeTFAlphaZeroProblem
    from rlmolecule.molecule.molecule_state import MoleculeState

    from tf_model import policy_model  #todo: this looks broken? (psj)
    import tf


    class StableRadOptProblem(MoleculeTFAlphaZeroProblem):

        def __init__(self,
                     engine: 'sqlalchemy.engine.Engine',
                     config: 'MoleculeConfig',
                     stability_model: 'tf.keras.Model',
                     redox_model: 'tf.keras.Model',
                     bde_model: 'tf.keras.Model',
                     **kwargs) -> None:
            super(StableRadOptProblem, self).__init__(engine, config, **kwargs)
            self.engine = engine
            self._config = config
            self.stability_model = stability_model
            self.redox_model = redox_model
            self.bde_model = bde_model

        def get_initial_state(self) -> MoleculeState:
            return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self._config)

        def get_reward(self, state: MoleculeState) -> Tuple[float, dict]:
            # TODO what does this do? Is it needed?
            ## Node is outside the domain of validity
            #elif ((node.policy_inputs['atom'] == 1).any() | 
            #    (node.policy_inputs['bond'] == 1).any()):
            #    return config.min_reward

            # TODO what data should be stored here?
            if state.forced_terminal:
                return self.calc_reward_inner(state), {'forced_terminal': True, 'smiles': state.smiles}
            return 0.0, {'forced_terminal': False, 'smiles': state.smiles}

        @tf.function(experimental_relax_shapes=True)                
        def predict(self, model: 'tf.keras.Model', inputs):
            return model.predict_step(inputs)

        def calc_reward_inner(self, state: MoleculeState) -> float:
            """
            """
            spins, buried_vol = self.predict(self.stability_model,
                {key: tf.constant(np.expand_dims(val, 0))
                for key, val in self.get_policy_inputs(state).items()})

            spins = spins.numpy().flatten()
            buried_vol = buried_vol.numpy().flatten()

            atom_index = int(spins.argmax())
            max_spin = spins[atom_index]
            spin_buried_vol = buried_vol[atom_index]

            atom_type = state.molecule.GetAtomWithIdx(atom_index).GetSymbol()

            ionization_energy, electron_affinity = self.predict(
                self.redox_model, {key: tf.constant(np.expand_dims(val, 0))
                            for key, val in self.get_policy_inputs(state).items()})\
                                                       .numpy().tolist()[0]

            v_diff = ionization_energy - electron_affinity
            bde, bde_diff = self.calc_bde(state)

            ea_range = (-.5, 0.2)
            ie_range = (.5, 1.2)
            v_range = (1, 1.7)   
            bde_range = (60, 80)    

            # This is a bit of a placeholder; but the range for spin is about 1/50th that
            # of buried volume.
            reward = ((1 - max_spin) * 50 + spin_buried_vol 
                    + 25 / (1 + np.exp(-(bde_diff - 10)))
                    + 100 * (
                self.windowed_loss(electron_affinity, ea_range) +     
                self.windowed_loss(ionization_energy, ie_range) + 
                self.windowed_loss(v_diff, v_range) +         
                self.windowed_loss(bde, bde_range)) / 4)

            return reward

        def calc_bde(self, state: MoleculeState):
            """calculate the X-H bde, and the difference to the next-weakest X-H bde in kcal/mol"""

            bde_inputs = prepare_for_bde(state.smiles)
            inputs = bde_get_inputs(bde_inputs.mol_smiles)

            pred_bdes = self.predict(self.bde_model, inputs)[0][0, :, 0].numpy()

            bde_radical = pred_bdes[bde_inputs.bond_index]

            if len(bde_inputs.other_h_bonds) == 0:
                bde_diff = 30.  # Just an arbitrary large number

            else:
                other_h_bdes = pred_bdes[bde_inputs.other_h_bonds]
                bde_diff = (other_h_bdes - bde_radical).min()

            return bde_radical, bde_diff

        def windowed_loss(
                self, target: float,
                desired_range: Tuple[float, float]) -> float:
            """ Returns 0 if the molecule is in the middle of the desired range,
            scaled loss otherwise. """

            span = desired_range[1] - desired_range[0]

            lower_lim = desired_range[0] + span / 6
            upper_lim = desired_range[1] - span / 6

            if target < lower_lim:
                return max(1 - 3*(abs(target - lower_lim) / span), 0)
            elif target > upper_lim:
                return max(1 - 3*(abs(target - upper_lim) / span), 0)
            else:
                return 1


    stability_model = tf.keras.models.load_model(stability_model, compile=False)
    redox_model = tf.keras.models.load_model(redox_model, compile=False)
    bde_model = tf.keras.models.load_model(bde_model, compile=False)

    # TODO what values should we set for these options?
    #:param max_atoms: Maximum number of heavy atoms
    #:param min_atoms: minimum number of heavy atoms
    # looks like atom_additions just used this default as well
    #:param atom_additions: potential atom types to consider. Defaults to ('C', 'H', 'O')
    #:param stereoisomers: whether to consider stereoisomers different molecules
    #:param sa_score_threshold: If set, don't construct molecules greater than a given sa_score.
    #:param tryEmbedding: Try to get a 3D embedding of the molecule, and if this fails, remote it.
    config = MoleculeConfig(max_atoms=25,
                            min_atoms=1,
                            tryEmbedding=True,
                            sa_score_threshold=4.,
                            stereoisomers=True)

    #engine = create_engine(f'sqlite:///stable_radical.db',
    #                       connect_args={'check_same_thread': False},
    #                       execution_options = {"isolation_level": "AUTOCOMMIT"})
    dbname = "bde"
    port = "5432"
    host = "yuma.hpc.nrel.gov"
    user = "rlops"
    # read the password from a file
    passwd_file = '/projects/rlmolecule/rlops_pass'
    with open(passwd_file, 'r') as f:
        passwd = f.read().strip()

    drivername = "postgresql+psycopg2"
    engine_str = f'{drivername}://{user}:{passwd}@{host}:{port}/{dbname}'
    engine = create_engine(engine_str, execution_options={"isolation_level": "AUTOCOMMIT"})

    run_id = 'stable_radical_optimization'

    reward_factory = RankedRewardFactory(
        engine=engine,
        run_id=run_id,
        reward_buffer_min_size=50,
        reward_buffer_max_size=250,
        ranked_reward_alpha=0.9
    )

    problem = StableRadOptProblem(
        engine,
        config,
        stability_model,
        redox_model,
        bde_model,
        run_id=run_id,
        reward_class=reward_factory,
        # TODO what values should we set for these options?
        features=8,
        num_heads=2,
        num_messages=1,
        min_buffer_size=15,
        policy_checkpoint_dir='policy_checkpoints'
    )

    return problem


def prepare_for_bde(smiles):

    mol = Chem.MolFromSmiles(smiles)
    radical_index = None
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetNumRadicalElectrons() != 0:
            assert radical_index == None
            is_radical = True
            radical_index = i

            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            atom.SetNumRadicalElectrons(0)
            break

    radical_rank = Chem.CanonicalRankAtoms(mol, includeChirality=True)[radical_index]

    mol_smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(mol_smiles)

    radical_index_reordered = list(Chem.CanonicalRankAtoms(
        mol, includeChirality=True)).index(radical_rank)

    molH = Chem.AddHs(mol)
    for bond in molH.GetAtomWithIdx(radical_index_reordered).GetBonds():
        if 'H' in {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}:
            bond_index = bond.GetIdx()
            break
        
    h_bond_indices = [bond.GetIdx() for bond in filter(
        lambda bond: ((bond.GetEndAtom().GetSymbol() == 'H') 
                      | (bond.GetBeginAtom().GetSymbol() == 'H')), molH.GetBonds())]
    
    other_h_bonds = list(set(h_bond_indices) - {bond_index})
            
    return pd.Series({
        'mol_smiles': mol_smiles,
        'radical_index_mol': radical_index_reordered,
        'bond_index': bond_index,
        'other_h_bonds': other_h_bonds
    })


def bde_get_inputs(mol_smiles):
    inputs = bde_preprocessor.construct_feature_matrices(mol_smiles, train=False)
    assert not (inputs['atom'] == 1).any() | (inputs['bond'] == 1).any()
    return {key: np.expand_dims(val, 0) for key, val in inputs.items()}


def run_games(**kwargs):
    from rlmolecule.alphazero.alphazero import AlphaZero
    game = AlphaZero(construct_problem(**kwargs))
    while True:
        path, reward = game.run(num_mcts_samples=50)
        #logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1][0]}')
        logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1][0]}')


def train_model(**kwargs):
    construct_problem(**kwargs).train_policy_model(steps_per_epoch=100,
                                           game_count_delay=20,
                                           verbose=2)


def setup_argparser():
    parser = argparse.ArgumentParser(
        description='Optimize stable radicals to work as both the anode and caathode of a redox-flow battery.')

    parser.add_argument('--train-policy', action="store_true", default=False,
                        help='Train the policy model only (on GPUs)')
    parser.add_argument('--rollout', action="store_true", default=False,
                        help='Run the game simulations only (on CPUs)')
    # '/projects/rlmolecule/pstjohn/models/20210214_radical_stability_new_data/',
    parser.add_argument('--stability-model', '-S', type=pathlib.Path, required=True,
                        help='Radical stability model for computing the electron spin and buried volume')
    # '/projects/rlmolecule/pstjohn/models/20210214_redox_new_data/',
    parser.add_argument('--redox-model', '-R', type=pathlib.Path, required=True,
                        help='Redox model for computing the ionization_energy and electron_affinity')
    # '/projects/rlmolecule/pstjohn/models/20210216_bde_new_nfp/',
    parser.add_argument('--bde-model', '-B', type=pathlib.Path, required=True,
                        help='BDE model for computing the Bond Dissociation Energy')

    return parser


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()
    kwargs = vars(args)

    if args.train_policy:
        train_model(**kwargs)
    elif args.rollout:
        run_games(**kwargs)
    else:
        print("Must specify either --train-policy or --rollout")
    # else:
    #     jobs = [multiprocessing.Process(target=monitor)]
    #     jobs[0].start()
    #     time.sleep(1)

    #     for i in range(5):
    #         jobs += [multiprocessing.Process(target=run_games)]

    #     jobs += [multiprocessing.Process(target=train_model)]

    #     for job in jobs[1:]:
    #         job.start()

    #     for job in jobs:
    #         job.join(300)
