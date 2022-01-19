""" Optimize the volume for the conducting ions
"""

import argparse
import logging
import math
import os
import time
import pandas as pd
## Apparently there's an issue with the latest version of pandas.
## Got this fix from here:
## https://github.com/pandas-profiling/pandas-profiling/issues/662#issuecomment-803673639
#pd.set_option("display.max_columns", None)
import random
import json
import gzip
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from pymatgen.core import Structure
from nfp import custom_objects
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor

from rlmolecule.crystal.builder import CrystalBuilder
from rlmolecule.crystal.crystal_problem import CrystalTFAlphaZeroProblem
from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.sql.run_config import RunConfig
# from rlmolecule.tree_search.reward import RankedRewardFactory
from rlmolecule.tree_search.reward import RankedRewardFactory
from rlmolecule.sql import Base, Session
from rlmolecule.sql.tables import GameStore
from scripts.nfp_extensions import RBFExpansion #, CifPreprocessor
from scripts import nrelmatdbtaps
from scripts import stability
from scripts import ehull

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_structures_file(structures_file):
    logger.info(f"reading {structures_file}")
    with gzip.open(structures_file, 'r') as f:
        structures_dict = json.loads(f.read().decode())
    structures = {}
    for key, structure_dict in structures_dict.items():
        structures[key] = Structure.from_dict(structure_dict)
    logger.info(f"\t{len(structures)} structures read")
    return structures


def write_structures_file(structures_file, structures_dict):
    """ Write Pymatgen structures to a gzipped json file. 
    *structures_dict*: dictionary of structure_id: dictionary representation of structure from `structure.as_dict()`
        See https://pymatgen.org/pymatgen.core.structure.html#pymatgen.core.structure.IStructure.as_dict
    """
    logger.info(f"writing {structures_file}")
    with gzip.open(structures_file, 'w') as out:
        out.write(json.dumps(structures_dict, indent=2).encode())


def generate_decoration(state: CrystalState, icsd_prototype) -> Structure:
    # Create the decoration of this composition onto this prototype structure
    # the 'action_node' string has the following format at this point:
    # comp_type|prototype_structure|decoration_idx
    # we just need 'comp_type|prototype_structure' to get the icsd structure
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
                 dist_model: 'tf.keras.Model',
                 df_competing_phases: 'pd.DataFrame',
                 # initial_state: str,
                 **kwargs) -> None:
        """ A class to estimate the suitability of a crystal structure as a solid state battery

        :param engine: A sqlalchemy engine pointing to a suitable database backend
        :param builder: A CrystalBuilder class to handle crystal construction based on ICSD structures
        :param energy_model: A tensorflow model to estimate the total energy of a structure
        :param dist_model: A tensorflow model to predict a binary value if the structure would 
            relax close (1) or far (0) from the starting prototype. 
            If close, the predicted energy should be more accurate.
        """
        # self.initial_state = initial_state
        self.engine = engine
        self.energy_model = energy_model
        self.dist_model = dist_model
        self.df_competing_phases = df_competing_phases
        # since the reward values can take positive or negative values, centered around 0,
        # set the default reward lower so that failed runs have a smaller reward
        self.default_reward = np.float64(-10)
        # if a structure is not predicted to relax close to the original prototype, penalize the reward value
        self.dist_penalty = -2
        super(CrystalEnergyStabilityOptProblem, self).__init__(engine, **kwargs)

    def get_reward(self, state: CrystalState) -> (float, {}):
        if not state.terminal:
            return self.default_reward, {'terminal': False, 'state_repr': repr(state)}
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
            decorated_structure = generate_decoration(state, icsd_prototype)
        except AssertionError as e:
            print(f"AssertionError: {e}")
            return self.default_reward, {'terminal': True, 'state_repr': repr(state)}

        # Predict the total energy and stability of this decorated structure
        predicted_energy, hull_energy = self.calc_energy_stability(decorated_structure)
        if predicted_energy is None:
            reward = self.default_reward - 2
            return reward, {'terminal': True,
                            'num_sites': len(icsd_prototype.sites),
                            'state_repr': repr(state)}
        if hull_energy is not None:
            # Since more negative is more stable, and higher is better for the reward values,
            # flip the hull energy
            reward = - hull_energy.astype(float)
        else:
            # subtract 1 to the default energy to distinguish between
            # failed calculation here, and failing to decorate the structure 
            reward = - self.default_reward - 1

        dist_pred = self.calc_cos_dist(decorated_structure)
        # TODO find the correct decision boundary to use. Should take the sigmoid, then apply the cutoff
        if dist_pred < 0:
            reward += self.dist_penalty

        # print(str(state), predicted_energy)
        info = {
            'terminal': True,
            'predicted_energy': predicted_energy.astype(float),
            'hull_energy': hull_energy.astype(float),
            'num_sites': len(decorated_structure.sites),
            'dist_pred': dist_pred.astype(float),
            'state_repr': repr(state),
        }
        return reward, info

    def get_model_inputs(self, structure, preprocessor) -> {}:
        inputs = preprocessor(structure, train=False)
        # scale structures to a minimum of 1A interatomic distance
        min_distance = inputs["distance"].min()
        if np.isclose(min_distance, 0):
            # if some atoms are on top of each other, then this normalization fails
            # TODO remove those problematic prototype structures
            return None
            #raise RuntimeError(f"Error with {structure}")

        inputs["distance"] /= inputs["distance"].min()
        return inputs

    # @collect_metrics
    def calc_energy_stability(self, structure: Structure, state=None):
        """ Predict the total energy of the structure using a GNN model (trained on unrelaxed structures)
        Then use that predicted energy to compute the decomposition (hull) energy
        """
        model_inputs = self.get_model_inputs(structure, preprocessor)
        if model_inputs is None:
            return None, None
        # predicted_energy = predict(self.energy_model, model_inputs)
        dataset = tf.data.Dataset.from_generator(
            lambda: (s for s in [model_inputs]),
            #lambda: (preprocessor.construct_feature_matrices(s, train=False) for s in [structure]),
            output_signature=(preprocessor.output_signature),
            )
        dataset = dataset \
            .padded_batch(
                batch_size=1,
                padding_values=(preprocessor.padding_values),
            )
        predicted_energy = self.energy_model.predict(dataset)
        predicted_energy = predicted_energy[0][0]

        comp = structure.composition.reduced_composition.alphabetical_formula.replace(' ','')
        hull_energy = ehull.convex_hull_stability(comp,
                                                  predicted_energy,
                                                  self.df_competing_phases)

        return predicted_energy, hull_energy

    def calc_cos_dist(self, structure: Structure, state=None):
        model_inputs = self.get_model_inputs(structure, preprocessor_dist)
        if model_inputs is None:
            return None
        dataset = tf.data.Dataset.from_generator(
            lambda: (s for s in [model_inputs]),
            #lambda: (preprocessor.construct_feature_matrices(s, train=False) for s in [structure]),
            output_signature=(preprocessor.output_signature),
            )
        dataset = dataset \
            .padded_batch(
                batch_size=1,
                padding_values=(preprocessor.padding_values),
            )
        predicted_dist = self.dist_model.predict(dataset)
        predicted_dist = predicted_dist[0][0]
        return predicted_dist


def create_problem():
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
                                               dist_model,
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
                                               )

    return problem


def run_games():
    from rlmolecule.alphazero.alphazero import AlphaZero

    prob_config = run_config.problem_config
    builder = CrystalBuilder(actions_to_ignore=prob_config.get('actions_to_ignore', None))

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
    # game = MCTS(
    #    create_problem(),
    # )
    # i = 0
    while True:
        path, reward = game.run(
            num_mcts_samples=config.get('num_mcts_samples', 5),
            max_depth=config.get('max_depth', 1000000),
        )
        logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1]}')

        # i += 1
        # if i % 10 == 0:
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
    parser.add_argument('--dist-model',
                        type=pathlib.Path,
                        required=True,
                        help='Model that predicts whether a given structure would relax '
                             'close or far from the starting prototype')

    args = parser.parse_args()

    run_config = RunConfig(args.config)
    run_id = run_config.run_id

    engine = run_config.start_engine()

    # Initialize the preprocessor class
    preprocessor = PymatgenPreprocessor()
    # TODO: shouldn't be hardcoded
    preprocessor.from_json('inputs/models/icsd_battery_relaxed/20211227_icsd_and_battery/preprocessor.json')
    preprocessor_dist = PymatgenPreprocessor()
    preprocessor_dist.from_json('inputs/models/cos_dist/model_b64_dist_class_0_1/preprocessor.json')

    # keep track of how much time each part takes
    # model_time = 0
    # decoration_time = 0

    energy_model = tf.keras.models.load_model(args.energy_model,
                                              custom_objects={**custom_objects,
                                                              **{'RBFExpansion': RBFExpansion}})
    dist_model = tf.keras.models.load_model(args.dist_model,
                                              custom_objects={**custom_objects,
                                                              **{'RBFExpansion': RBFExpansion}})
    # Dataframe containing competing phases from NRELMatDB
    print("Reading inputs/competing_phases.csv")
    df_competing_phases = pd.read_csv('inputs/competing_phases.csv')
    print(df_competing_phases.head(2))

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
