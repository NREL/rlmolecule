""" Optimize crystals for thermodynamic stability
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
from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor
from nfp import custom_objects
from nfp.layers import RBFExpansion
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from pymatgen.core.periodic_table import Element

from rlmolecule.crystal.builder import CrystalBuilder
from rlmolecule.crystal.crystal_problem import CrystalTFAlphaZeroProblem
from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.sql.run_config import RunConfig
# from rlmolecule.tree_search.reward import RankedRewardFactory
from rlmolecule.tree_search.reward import RankedRewardFactory
from rlmolecule.sql import Base, Session
from rlmolecule.sql.tables import GameStore, RewardStore

from reward import CrystalReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AtomicNumberPreprocessor(PymatgenPreprocessor):
    def __init__(self, max_atomic_num=83, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site_tokenizer = lambda x: Element(x).Z
        self._max_atomic_num = max_atomic_num

    @property
    def site_classes(self):
        return self._max_atomic_num


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
                 rewarder: CrystalReward,
                 # initial_state: str,
                 **kwargs) -> None:
        """ A class to estimate the suitability of a crystal structure as a solid state battery

        :param engine: A sqlalchemy engine pointing to a suitable database backend
        :param builder: A CrystalBuilder class to handle crystal construction based on ICSD structures
        """
        # self.initial_state = initial_state
        self.engine = engine
        self.rewarder = rewarder
        # since the reward values can take positive or negative values, centered around 0,
        # set the default reward lower so that failed runs have a smaller reward
        self.default_reward = np.float64(-5)
        super(CrystalEnergyStabilityOptProblem, self).__init__(engine, **kwargs)

    def get_reward(self, state: CrystalState) -> (float, {}):
        return self.rewarder.get_reward(state)


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

    rewarder = CrystalReward(prototype_structures,
                             energy_model,
                             preprocessor,
                             df_competing_phases,
                             vol_pred_site_bias=site_bias)

    problem = CrystalEnergyStabilityOptProblem(engine,
                                               rewarder,
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
    builder = CrystalBuilder(G=prob_config.get('action_graph1'),
                             G2=prob_config.get('action_graph2'),
                             actions_to_ignore=prob_config.get('actions_to_ignore'))

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
    i = 0
    states_seen = set()
    # remove deadends before starting the rollouts
    set_states_seen(game, states_seen)
    while True:
        path, reward = game.run(
            num_mcts_samples=config.get('num_mcts_samples', 5),
            max_depth=config.get('max_depth', 1000000),
        )
        logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1]}')

        if i % 2 == 0:
            set_states_seen(game, states_seen)
        i += 1


def set_states_seen(game, states_seen):
    start = time.process_time()
    # UPDATE 2022-03-01:
    # load the crystal states with a computed reward,
    # and add them to the states_to_ignore so they will not be
    # available as part of the search space anymore
    df = pd.read_sql(session.query(RewardStore)
                            .filter_by(run_id=run_config.run_id)
                            .statement, session.bind)
    # only keep the terminal states, and those with a reward value above a cutoff
    # Since states with low reward values won't be repeatedly selected,
    # the reward cutoff should be able to be relatively high e.g., > 0
    reward_cutoff = -.5
    df['state'] = df['data'].apply(lambda x: x['state_repr'])
    df['terminal'] = df['data'].apply(lambda x: str(x['terminal']).lower() == "true")
    states = set(df[(df['terminal']) & (df['reward'] > reward_cutoff)]['state'].values)
    for state in states - states_seen:
        comp = state.split('|')[0]
        decoration_node = '|'.join(state.split('|')[1:])
        game.state_builder.states_to_ignore[comp].add(decoration_node)

    states_seen |= states
    #print(f"{len(states - states_seen)} new states since last checked. "
    #      f"len(states_seen): {len(states_seen)}")
    # check to make sure there are no dead-ends e.g., compositions with no prototypes available
    game.state_builder.remove_dead_ends()
    print(f"{len(game.state_builder.actions_to_ignore)}, "
          f"{len(game.state_builder.actions_to_ignore_G2)}, "
          f"{len(game.state_builder.states_to_ignore)} "
          "actions to ignore in G, G2, and states, respectively")
    print(f"\t took {time.process_time() - start:0.2f} sec")
    return states_seen


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
                  f"{best_reward.data['state']} with {num_games} games played")

        time.sleep(5)


# load the icsd prototype structures
# https://pymatgen.org/usage.html#side-note-as-dict-from-dict
icsd_prototypes_file = "../../rlmolecule/crystal/inputs/icsd_prototypes_lt50atoms_lt100dist.json.gz"
prototype_structures = read_structures_file(icsd_prototypes_file)

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
    parser.add_argument('--vol-pred-site-bias', type=pathlib.Path,
                        help='Apply a volume prediction to the decorated structure '
                        'before passing it to the GNN. '
                        'Give the path to a file with the average volume per element')

    args = parser.parse_args()

    run_config = RunConfig(args.config)
    run_id = run_config.run_id

    engine = run_config.start_engine()

    # Initialize the preprocessor class
    #preprocessor = AtomicNumberPreprocessor()
    preprocessor = PymatgenPreprocessor()
    preprocessor_file = os.path.dirname(args.energy_model) + '/preprocessor.json'
    preprocessor.from_json(preprocessor_file)

    energy_model = tf.keras.models.load_model(args.energy_model,
                                              custom_objects={**custom_objects,
                                                              **{'RBFExpansion': RBFExpansion}})
    # Dataframe containing competing phases from NRELMatDB
    print("Reading inputs/competing_phases.csv")
    df_competing_phases = pd.read_csv('inputs/competing_phases.csv')
    print(f"\t{len(df_competing_phases)} lines")
    print(df_competing_phases.head(2))

    site_bias = None
    if args.vol_pred_site_bias is not None:
        print(f"Reading {args.vol_pred_site_bias}")
        site_bias = pd.read_csv(args.vol_pred_site_bias,
                                index_col=0, squeeze=True)
        print(f"\t{len(site_bias)} elements")

    # map the elements in elements_to_skip to the corresponding action nodes (tuples) 
    # that have those elements
    prob_config = run_config.problem_config
    if 'elements_to_ignore' in prob_config:
        eles_to_ignore = prob_config['elements_to_ignore']
        actions_to_ignore = set(prob_config.get('actions_to_ignore', []))
        num_acts = len(actions_to_ignore)
        builder = CrystalBuilder()
        for ele in eles_to_ignore:
            for n in builder.G.nodes():
                if isinstance(n, tuple) and ele in n:
                    actions_to_ignore.add(n)
        print(f"{len(eles_to_ignore)} elements to ignore mapped to "
              f"{len(actions_to_ignore) - num_acts} new actions to ignore")
        prob_config['actions_to_ignore'] = actions_to_ignore

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
