""" Optimize the volume for the conducting ions
"""

import argparse
import logging
import math
import os
import time
import pandas as pd
import networkx as nx
import random
import json
import gzip
from collections import defaultdict
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
    logger.info(f"writing {structures_file}")
    with gzip.open(structures_file, 'w') as out:
        out.write(json.dumps(structures_dict, indent=2).encode())


def compute_structure_vol(structure: Structure, state=None):
    """ compute the total volume and the volume of just the conducting ions
    """
    # if the voronoi search fails, could try increasing the cutoff here
    for nn in [nn13]:
        try:
            voronoi_stats = nn.get_all_voronoi_polyhedra(structure)
            break
        # this function often fails for large or spaced out structures
        except ValueError as e:
            if state:
                logger.warning(f"compute_structure_vol:ValueError: {e}  -  {state}")
            return 0, 0
        except MemoryError as e:
            if state:
                logger.warning(f"compute_structure_vol:MemoryError: {e}  -  {state}")
            return 0, 0
        except RuntimeError as e:
            if state:
                logger.warning(f"compute_structure_vol:RuntimeError: {e}  -  {state}")
            return 0, 0

    total_vol = 0
    conducting_ion_vol = defaultdict(int)
    for atom in voronoi_stats:
        for site, site_info in atom.items():
            vol = site_info['volume']
            total_vol += vol

            element = site_info['site'].as_dict()['species'][0]['element']
            if element in conducting_ions:
                conducting_ion_vol[element] += vol

    # Zn can be either a conducting ion or a framework cation.
    # Make sure we're counting it correctly here
    if len(conducting_ion_vol) == 1:
        conducting_ion_vol = list(conducting_ion_vol.values())[0]
    elif len(conducting_ion_vol) == 2:
        # remove Zn
        correct_ion = list(set(conducting_ion_vol.keys()) - {'Zn'})[0]
        conducting_ion_vol = conducting_ion_vol[correct_ion]
    else:
        logger.warning(f"Expected 1 conducting ion. Found {len(conducting_ion_vol)}")
        conducting_ion_vol = 0

    return conducting_ion_vol, total_vol


class CrystalVolOptimizationProblem(CrystalTFAlphaZeroProblem):
    # def get_initial_state(self) -> CrystalState:
    #    return CrystalState(rdkit.Chem.MolFromSmiles('C'), self._config)

    def get_reward(self, state: CrystalState) -> (float, {}):
        if state.terminal:
            # Create the decoration of this composition onto this prototype structure
            # the 'action_node' string has the following format at this point:
            # comp_type|prototype_structure|decoration_idx
            # we just need 'comp_type|prototype_structure' to get the icsd structure
            structure_key = '|'.join(state.action_node.split('|')[:-1])
            icsd_prototype = structures[structure_key]
            decoration_idx = int(state.action_node.split('|')[-1]) - 1
            try:
                decorated_structure, comp = CrystalState.decorate_prototype_structure(
                    icsd_prototype, state.composition, decoration_idx=decoration_idx)
            except AssertionError as e:
                print(f"AssertionError: {e}")
                return 0.0, {'terminal': True, 'state_repr': repr(state)}

            # Compute the volume of the conducting ions.
            conducting_ion_vol, total_vol = compute_structure_vol(decorated_structure, state=str(state))
            frac_conducting_ion_vol = conducting_ion_vol / total_vol if total_vol != 0 else 0
            info = {
                'terminal': True,
                'conducting_ion_vol': conducting_ion_vol,
                'total_vol': total_vol,
                'state_repr': repr(state),
            }
            return frac_conducting_ion_vol, info
        return 0.0, {'terminal': False, 'state_repr': repr(state)}


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

    problem = CrystalVolOptimizationProblem(engine,
                                            run_id=run_id,
                                            reward_class=reward_factory,
                                            features=train_config.get('features', 64),
                                            max_buffer_size=train_config.get('max_buffer_size', 200),
                                            min_buffer_size=train_config.get('min_buffer_size', 15),
                                            batch_size=train_config.get('batch_size', 32),
                                            policy_checkpoint_dir=train_config.get('policy_checkpoint_dir',
                                                                                  'policy_checkpoints'))

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

        # i += 1
        # if i % 10 == 0:
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


# want to maximize the volume around only the conducting ions
conducting_ions = set(['Li', 'Na', 'K', 'Mg', 'Zn'])
anions = set(['F', 'Cl', 'Br', 'I', 'O', 'S', 'N', 'P'])
framework_cations = set(
    ['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'W', 'Zn', 'Cd', 'Hg', 'B', 'Al', 'Si', 'Ge', 'Sn', 'P', 'Sb'])

# Many structures fail with the default cutoff radius in Angstrom to look for near-neighbor atoms (13.0)
# with the error: "No Voronoi neighbors found for site".
# see: https://github.com/materialsproject/pymatgen/blob/v2022.0.8/pymatgen/analysis/local_env.py#L639.
# Increasing the cutoff takes longer. If I bump it up to 1000, it can take over 100 Gb of Memory!
nn13 = local_env.VoronoiNN(cutoff=13, compute_adj_neighbors=False)

# also load the icsd prototype structures
# https://pymatgen.org/usage.html#side-note-as-dict-from-dict
icsd_prototypes_file = "../../rlmolecule/crystal/inputs/icsd_prototypes.json.gz"
structures = read_structures_file(icsd_prototypes_file)

# # Temporary caching approach:
# # store the computed structures in a json file
# decorations = {}
# run_id = "2021-07-01"
# # base_dir = f"/projects/rlmolecule/jlaw/crystals/{run_id}"
# base_dir = f"./{run_id}"
# os.makedirs(base_dir, exist_ok=True)
# decorations_file = f"{base_dir}/decorations.json.gz"
# if os.path.isfile(decorations_file):
#     decorations = read_structures_file(decorations_file)
# # keep track of the composition volume and total volume for each decoration
# volume_stats = {}
# errored = set()
# # shubham said they didn't consider prototypes with more than 50 atoms when calculating the DFT relaxed structures.
# max_atoms = 50
# skipped_max_atoms = []
# skipped_prototypes = set()
# out_file = f"{base_dir}/volume-stats.tsv"
# # if os.path.isfile(out_file):
# #     logger.info(f"reading {out_file}")
# #     df = pd.read_csv(out_file, sep='\t', index_col=0)
# #     volume_stats = dict(zip(df.index, df.values))
# #     logger.info(f"\tread {len(volume_stats)} values")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the crystal conduction ion volume optimization. ' +
                                                 'Default is to run the script locally')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--train-policy', action="store_true", default=False,
                        help='Train the policy model only (on GPUs)')
    parser.add_argument('--rollout', action="store_true", default=False, help='Run the game simulations only (on CPUs)')

    args = parser.parse_args()

    run_config = RunConfig(args.config)
    run_id = run_config.run_id

    engine = run_config.start_engine()
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
