""" Optimize the volume for the conducting ions
"""

import argparse
import logging
import math
import os
import time
import pandas as pd
# Apparently there's an issue with the latest version of pandas. 
# Got this fix from here:
# https://github.com/pandas-profiling/pandas-profiling/issues/662#issuecomment-803673639
pd.set_option("display.max_columns", None)
import networkx as nx
import random
import ujson
import gzip
import pathlib
from collections import defaultdict
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import nfp
from pymatgen.core import Structure
from pymatgen.analysis import local_env

from rlmolecule.crystal.builder import CrystalBuilder
from rlmolecule.crystal.crystal_problem import CrystalTFAlphaZeroProblem
from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.sql.run_config import RunConfig
# from rlmolecule.tree_search.reward import RankedRewardFactory
from rlmolecule.tree_search.reward import LinearBoundedRewardFactory, RankedRewardFactory
from rlmolecule.sql import Base, Session
from rlmolecule.sql.tables import GameStore
from scripts.nfp_extensions import RBFExpansion, CifPreprocessor
from scripts import nrelmatdbtaps
from scripts import stability
from scripts import ehull

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_structures_file(structures_file):
    logger.info(f"reading {structures_file}")
    with gzip.open(structures_file, 'r') as f:
        structures_dict = ujson.loads(f.read().decode())
    structures = {}
    for key, structure_dict in structures_dict.items():
        structures[key] = Structure.from_dict(structure_dict)
    logger.info(f"\t{len(structures)} structures read")
    return structures


def write_structures_file(structures_file, structures_dict):
    logger.info(f"writing {structures_file}")
    with gzip.open(structures_file, 'w') as out:
        out.write(ujson.dumps(structures_dict, indent=2).encode())


def generate_decoration(state: CrystalState) -> Structure:
    # Create the decoration of this composition onto this prototype structure
    # the 'action_node' string has the following format at this point:
    # comp_type|prototype_structure|decoration_idx
    # we just need 'comp_type|prototype_structure' to get the icsd structure
    structure_key = '|'.join(state.action_node.split('|')[:-1])
    icsd_prototype = structures[structure_key]
    decoration_idx = int(state.action_node.split('|')[-1]) - 1
    decorated_structure, stoich = CrystalState.decorate_prototype_structure(
        icsd_prototype, state.composition, decoration_idx=decoration_idx)
    return decorated_structure


@tf.function(experimental_relax_shapes=True)
def predict(model: 'tf.keras.Model', inputs):
    return model.predict_step(inputs)


class CrystalEnergyStabilityOptProblem(CrystalTFAlphaZeroProblem):
    def __init__(self,
                 engine: 'sqlalchemy.engine.Engine',
                 energy_model: 'tf.keras.Model',
                 df_competing_phases: 'pd.DataFrame', 
                 #initial_state: str,
                 **kwargs) -> None:
        """ A class to estimate the suitability of a crystal structure as a solid state battery

        :param engine: A sqlalchemy engine pointing to a suitable database backend
        :param builder: A CrystalBuilder class to handle crystal construction based on ICSD structures
        :param energy_model: A tensorflow model to estimate the total energy of a structure
        """
        #self.initial_state = initial_state
        self.engine = engine
        self.energy_model = energy_model
        self.df_competing_phases = df_competing_phases
        # since the reward values can take positive or negative values, centered around 0,
        # set the default reward lower so that failed runs have a smaller reward
        self.default_reward = -5
        super(CrystalEnergyStabilityOptProblem, self).__init__(engine, **kwargs)

    def get_reward(self, state: CrystalState) -> (float, {}):
        if state.terminal:
            # skip this structure if it is too large for the model
            # TODO truncate the structure?
            structure_key = '|'.join(state.action_node.split('|')[:-1])
            icsd_prototype = structures[structure_key]
            if len(icsd_prototype.sites) > 150:
                return self.default_reward, {'terminal': True,
                             'num_sites': len(icsd_prototype.sites),
                             'state_repr': repr(state)}

            # generate the decoration for this state
            try:
                decorated_structure = generate_decoration(state)
            except AssertionError as e:
                print(f"AssertionError: {e}")
                return self.default_reward, {'terminal': True, 'state_repr': repr(state)}
            predicted_energy, hull_energy = self.calc_energy_stability(decorated_structure)
            #print(str(state), predicted_energy)
            # Predict the total energy and stability of this decorated structure
            info = {
                'terminal': True,
                'predicted_energy': predicted_energy.astype(float),
                'hull_energy': hull_energy.astype(float),
                'num_sites': len(decorated_structure.sites),
                'state_repr': repr(state),
            }
            #return stability, info
            # since any hull energy is better than nothing, add 10 to all the energies
            # (with the hull energy flipped since more negative is more stable)
            return - hull_energy.astype(float), info
        return self.default_reward, {'terminal': False, 'state_repr': repr(state)}

    def get_model_inputs(self, structure) -> {}:
        inputs = preprocessor.construct_feature_matrices(structure, train=False)
        print(inputs)
        #return {key: tf.constant(np.expand_dims(val, 0)) for key, val in inputs.items()}
        return inputs

    #@collect_metrics
    def calc_energy_stability(self, structure: Structure, state=None):
        """ Predict the total energy of the structure using a GNN model (trained on unrelaxed structures)
        """
        #model_inputs = self.get_model_inputs(structure)
        #predicted_energy = predict(self.energy_model, model_inputs)
        dataset = tf.data.Dataset.from_generator(
            #lambda: preprocessor.construct_feature_matrices(structure, train=False),
            lambda: (preprocessor.construct_feature_matrices(s, train=False) for s in [structure]),
            output_types=preprocessor.output_types,
            output_shapes=preprocessor.output_shapes)\
            .padded_batch(batch_size=32,
                          padded_shapes=preprocessor.padded_shapes(max_sites=256, max_bonds=2048),
                          padding_values=preprocessor.padding_values)
        predicted_energy = self.energy_model.predict(dataset)
        predicted_energy = predicted_energy[0][0]

        hull_energy = self.convex_hull_stability(structure, predicted_energy)
        if hull_energy is None:
            # set the default hull energy as slightly bigger than the default energy
            hull_energy = -self.default_reward - 1

        return predicted_energy, hull_energy

    def convex_hull_stability(self, structure: Structure, predicted_energy):
        strc = structure

        # Add the new composition and the predicted energy to "df" if DFT energy already not present
        comp = strc.composition.reduced_composition.alphabetical_formula.replace(' ','')

        df = self.df_competing_phases
        if comp not in df.reduced_composition.tolist():
            df = self.df_competing_phases.append({'sortedformula': comp, 'energyperatom': predicted_energy, 'reduced_composition': comp}, ignore_index=True)

        # Create a list of elements in the composition
        ele = strc.composition.chemical_system.split('-')

        # Create input file for stability analysis 
        inputs = nrelmatdbtaps.create_input_DFT(ele, df, chempot='ferev2')

        # Run stability function (args: input filename, composition)
        stable_state = stability.run_stability(inputs, comp)
        if stable_state == 'UNSTABLE':
            stoic = ehull.frac_stoic(comp)
            hull_nrg = ehull.unstable_nrg(stoic, comp, inputs)
            #print("energy above hull of this UNSTABLE phase is", hull_nrg, "eV/atom")
        elif stable_state == 'STABLE':
            stoic = ehull.frac_stoic(comp)
            hull_nrg = ehull.stable_nrg(stoic, comp, inputs)
            #print("energy above hull of this STABLE phase is", hull_nrg, "eV/atom")
        else:
            print(f"ERR: unrecognized stable_state: '{stable_state}'")
        return hull_nrg

    ## TODO
    #@collect_metrics
    #def calc_reward(self, state: CrystalState) -> (float, {}):
    #    """
    #    """
    #    reward = 0
    #    stats = {}
    #    return reward, stats


def create_problem():
    prob_config = run_config.problem_config

    run_id = run_config.run_id
    train_config = run_config.train_config

    reward_factory = RankedRewardFactory(engine=engine,
                                         run_id=run_id,
                                         reward_buffer_min_size=train_config.get('reward_buffer_min_size', 10),
                                         reward_buffer_max_size=train_config.get('reward_buffer_max_size', 50),
                                         ranked_reward_alpha=train_config.get('ranked_reward_alpha', 0.75))

    # reward_factory = LinearBoundedRewardFactory(min_reward=train_config.get('min_reward', 0),
    #                                             max_reward=train_config.get('max_reward', 1))

    problem = CrystalEnergyStabilityOptProblem(engine,
                                               energy_model,
                                               df_competing_phases,
                                               run_id=run_id,
                                               reward_class=reward_factory,
                                               features=train_config.get('features', 64),
                                               num_heads=train_config.get('num_heads', 4),
                                               num_messages=train_config.get('num_messages', 3),
                                               max_buffer_size=train_config.get('max_buffer_size', 200),
                                               min_buffer_size=train_config.get('min_buffer_size', 15),
                                               batch_size=train_config.get('batch_size', 32),
                                               policy_checkpoint_dir=train_config.get('policy_checkpoint_dir',
                                                                                  'policy_checkpoints'),
                                               actions_to_ignore=prob_config.get('actions_to_ignore', None),
                                               )

    return problem


def run_games():
    from rlmolecule.alphazero.alphazero import AlphaZero

    builder = CrystalBuilder()
    config = run_config.mcts_config
    game = AlphaZero(
        create_problem(),
        min_reward=config.get('min_reward', 0.0),
        pb_c_base=config.get('pb_c_base', 1.0),
        pb_c_init=config.get('pb_c_init', 1.25),
        dirichlet_noise=config.get('dirichlet_noise', True),
        dirichlet_alpha=config.get('dirichlet_alpha', 1.0),
        dirichlet_x=config.get('dirichlet_x', 0.25),
        # MCTS parameters
        ucb_constant=config.get('ucb_constant', math.sqrt(2)),
        state_builder=builder,
    )
    from rlmolecule.mcts.mcts import MCTS
    #game = MCTS(
    #    create_problem(),
    #)
    # i = 0
    while True:
        path, reward = game.run(
            num_mcts_samples=config.get('num_mcts_samples', 5),
            max_depth=config.get('max_depth', 1000000),
        )
        logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1]}')

        #i += 1
        #if i % 10 == 0:
        #    print(f"decoration_time: {decoration_time}, model_time: {model_time}")
        #     df = pd.DataFrame(volume_stats).T
        #     df.columns = ['conducting_ion_vol', 'total_vol', 'fraction', 'comp_type']
        #     df = df.sort_index()
        #     print(f"writing current stats to {out_file}")
        #     df.to_csv(out_file, sep='\t')
            # write_structures_file(decorations_file, decorations)


def train_model():
    config = run_config.train_config
    create_problem().train_policy_model(steps_per_epoch=config.get('steps_per_epoch', 100),
                                        lr=float(config.get('lr', 1E-3)),
                                        epochs=int(float(config.get('epochs', 1E4))),
                                        game_count_delay=config.get('game_count_delay', 20),
                                        verbose=config.get('verbose', 2))


# TODO copied from alphazero_problem.py
def iter_recent_games():
    """Iterate over randomly chosen positions in games from the replay buffer

    :returns: a generator of (serialized_parent, visit_probabilities, scaled_reward) pairs
    """

    recent_games = session.query(GameStore).filter_by(run_id=run_id) \
        .order_by(GameStore.time.desc()).limit(200)

    for game in recent_games:
        parent_state_string, visit_probabilities = random.choice(game.search_statistics)
        policy_digests, visit_probs = zip(*visit_probabilities)
        yield ([parent_state_string] + list(policy_digests), [game.scaled_reward] + list(visit_probs))


def monitor():
    from rlmolecule.sql.tables import RewardStore
    problem = create_problem()

    while True:
        # best_reward = problem.session.query(RewardStore) \
        # .filter_by(run_id=problem.run_id) \
        best_reward = session.query(RewardStore) \
            .filter_by(run_id=run_id) \
            .order_by(RewardStore.reward.desc()).first()

        num_games = len(list(iter_recent_games()))
        print(best_reward, num_games)

        if best_reward:
            print(f"Best Reward: {best_reward.reward:.3f} for molecule "
                  f"{best_reward.data['smiles']} with {num_games} games played")

        time.sleep(5)


# load the icsd prototype structures
# https://pymatgen.org/usage.html#side-note-as-dict-from-dict
icsd_prototypes_file = "../../rlmolecule/crystal/inputs/icsd_prototypes.json.gz"
structures = read_structures_file(icsd_prototypes_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the battery structure stable energy optimization. ' +
                                                 'Default is to run the script locally')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--train-policy', action="store_true", default=False,
                        help='Train the policy model only (on GPUs)')
    parser.add_argument('--rollout', action="store_true", default=False, help='Run the game simulations only (on CPUs)')
    parser.add_argument('--energy-model',
                        type=pathlib.Path,
                        required=True,
                        help='Model for predicting total energy of a battery system')

    args = parser.parse_args()

    run_config = RunConfig(args.config)
    run_id = run_config.run_id

    engine = run_config.start_engine()

    # Initialize the preprocessor class
    preprocessor = CifPreprocessor(num_neighbors=12)
    preprocessor.from_json('inputs/preprocessor.json')

    # keep track of how much time each part takes
    #model_time = 0
    #decoration_time = 0

    energy_model = tf.keras.models.load_model(args.energy_model,
                                              custom_objects={**nfp.custom_objects, **{'RBFExpansion': RBFExpansion}})
    # Dataframe containing competing phases from NRELMatDB
    print("Reading inputs/competing_phases.csv")
    df_competing_phases = pd.read_csv('inputs/competing_phases.csv')
    print(df_competing_phases.head(3))

    Base.metadata.create_all(engine, checkfirst=True)
    Session.configure(bind=engine)
    session = Session()

    if args.train_policy:
        train_model()
    if args.rollout:
        # make sure the rollouts do not use the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        run_games()
    else:
        monitor()
        # jobs = [multiprocessing.Process(target=monitor)]
        # jobs[0].start()
        # time.sleep(1)
        #
        # for i in range(5):
        #     jobs += [multiprocessing.Process(target=run_games)]
        #
        # jobs += [multiprocessing.Process(target=train_model)]
        #
        # for job in jobs[1:]:
        #     job.start()
        #
        # start = time.time()
        # while time.time() - start <= run_config.problem_config.get('timeout', 300):
        #     time.sleep(1)
        #
        # for j in jobs:
        #     j.terminate()
        #     j.join()
