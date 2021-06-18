""" Optimize the volume for the conducting ions
"""

import argparse
import logging
import math
import multiprocessing
import os
import time
import pandas as pd
import networkx as nx
import random

from examples.crystal_volume.builder import CrystalBuilder
#from examples.crystal_volume.crystal_problem import CrystalTFAlphaZeroProblem
from examples.crystal_volume.crystal_problem import CrystalProblem
from examples.crystal_volume.crystal_state import CrystalState
from rlmolecule.sql.run_config import RunConfig
from rlmolecule.tree_search.reward import RankedRewardFactory
from rlmolecule.sql import Base, Session, digest, load_numpy_dict, serialize_ordered_numpy_dict
from rlmolecule.sql.tables import GameStore, RewardStore, StateStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run the crystal conduction ion volume optimization. ' +
                                             'Default is to run the script locally')
parser.add_argument('--config', type=str, help='Configuration file')
parser.add_argument('--train-policy', action="store_true", default=False, help='Train the policy model only (on GPUs)')
parser.add_argument('--rollout', action="store_true", default=False, help='Run the game simulations only (on CPUs)')

args = parser.parse_args()

run_config = RunConfig(args.config)
run_id = run_config.run_id

engine = run_config.start_engine()
Base.metadata.create_all(engine, checkfirst=True)
Session.configure(bind=engine)
session = Session()


class CrystalVolOptimizationProblem(CrystalProblem):
    #def get_initial_state(self) -> CrystalState:
    #    return CrystalState(rdkit.Chem.MolFromSmiles('C'), self._config)

    def get_reward(self, state: CrystalState) -> (float, {}):
        #if state.forced_terminal:
        #    return qed(state.molecule), {'forced_terminal': True, 'smiles': state.smiles}
        if state.terminal:
            # TODO generate the pymatgen structure object for this decoration of the crystal structure
            # and compute the volume of the conducting ions.
            # For now, set a dummy reward which is the length of the string representation
            return len(repr(state)), {'terminal': True, 'state_repr': repr(state)}
        return 0.0, {'terminal': False, 'state_repr': repr(state)}


def create_problem():
    prob_config = run_config.problem_config

    G = nx.DiGraph()
    G2 = nx.DiGraph()
    action_graph_file = "inputs/elements_to_compositions.edgelist.gz"
    print(f"reading {action_graph_file}")
    G = nx.read_edgelist(action_graph_file, delimiter='\t', data=False, create_using=G)
    print(f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    action_graph2_file = "inputs/comp_type_to_decorations.edgelist.gz"
    print(f"reading {action_graph2_file}")
    G2 = nx.read_edgelist(action_graph2_file, delimiter='\t', data=False, create_using=G2)
    print(f'{G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges')

    # also load the mapping from composition to composition type
    working_dir = 'inputs'
    df_comp = pd.read_csv(f'{working_dir}/compositions.csv.gz')
    print(df_comp.head())
    compositions = df_comp['composition'].to_list()
    comp_types = set(df_comp['comp_type'].to_list())
    comp_to_comp_type = dict(zip(df_comp['composition'], df_comp['comp_type']))

    builder = CrystalBuilder(G,
                             G2,
                             comp_to_comp_type,
    )

    run_id = run_config.run_id
    train_config = run_config.train_config

    reward_factory = RankedRewardFactory(engine=engine,
                                         run_id=run_id,
                                         reward_buffer_min_size=train_config.get('reward_buffer_min_size', 10),
                                         reward_buffer_max_size=train_config.get('reward_buffer_max_size', 50),
                                         ranked_reward_alpha=train_config.get('ranked_reward_alpha', 0.75))

    problem = CrystalVolOptimizationProblem(#engine,
                                     builder,
                                     #run_id=run_id,
                                     reward_class=reward_factory,
                                     #num_messages=train_config.get('num_messages', 1),
                                     #num_heads=train_config.get('num_heads', 2),
                                     #features=train_config.get('features', 8),
                                     #max_buffer_size=train_config.get('max_buffer_size', 200),
                                     #min_buffer_size=train_config.get('min_buffer_size', 15),
                                     #batch_size=train_config.get('batch_size', 32),
                                     #policy_checkpoint_dir=train_config.get('policy_checkpoint_dir',
                                     #                                       'policy_checkpoints'))
                                            )

    return problem


def run_games():
    # from rlmolecule.alphazero.alphazero import AlphaZero
    config = run_config.mcts_config
    # game = AlphaZero(
    #     create_problem(),
    #     min_reward=config.get('min_reward', 0.0),
    #     pb_c_base=config.get('pb_c_base', 1.0),
    #     pb_c_init=config.get('pb_c_init', 1.25),
    #     dirichlet_noise=config.get('dirichlet_noise', True),
    #     dirichlet_alpha=config.get('dirichlet_alpha', 1.0),
    #     dirichlet_x=config.get('dirichlet_x', 0.25),
    #     # MCTS parameters
    #     ucb_constant=config.get('ucb_constant', math.sqrt(2)),
    # )
    from rlmolecule.mcts.mcts import MCTS
    game = MCTS(
        create_problem(),
    )
    while True:
        path, reward = game.run(
            num_mcts_samples=config.get('num_mcts_samples', 50),
            max_depth=config.get('max_depth', 1000000),
        )
        logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1]}')


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
        #best_reward = problem.session.query(RewardStore) \
            #.filter_by(run_id=problem.run_id) \
        best_reward = session.query(RewardStore) \
            .filter_by(run_id=run_id) \
            .order_by(RewardStore.reward.desc()).first()

        num_games = len(list(iter_recent_games()))
        print(best_reward, num_games)

        if best_reward:
            print(f"Best Reward: {best_reward.reward:.3f} for molecule "
                  f"{best_reward.data['smiles']} with {num_games} games played")

        time.sleep(5)


if __name__ == "__main__":

    if args.train_policy:
        print("policy model not yet implemented")
    #     train_model()
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
