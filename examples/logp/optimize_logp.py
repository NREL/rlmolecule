""" Optimize the logP of a molecule
Starting point: methane (C)
  - actions: add a bond or an atom
  - state: molecule state
  - reward: 0, unless a terminal state is reached, then the penalized logp estimate of the molecule
"""

import argparse
import logging
import math
import multiprocessing
import os
import sys
import time
import networkx as nx

import rdkit
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors
#from rdkit.Contrib import SA_Score
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# noinspection PyUnresolvedReferences
import sascorer

from rlmolecule.sql.run_config import RunConfig
from examples.qed.optimize_qed import construct_problem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# copied from here: https://github.com/google-research/google-research/blob/master/mol_dqn/experimental/optimize_logp.py
# Zhou et al., Optimization of Molecules via Deep Reinforcement Learning. Scientific Reports 2019
def num_long_cycles(mol):
    """Calculate the number of long cycles.
    Args:
      mol: Molecule. A molecule.
    Returns:
      negative cycle length.
    """
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return -cycle_length


def penalized_logp(molecule):
  log_p = Descriptors.MolLogP(molecule)
  sas_score = sascorer.calculateScore(molecule)
  cycle_score = num_long_cycles(molecule)
  return log_p - sas_score + cycle_score


# copied from here: https://github.com/dbkgroup/prop_gen/blob/d17d935a534b6a667d2603b4d0c7b4add446d6bf/gym-molecule/gym_molecule/envs/molecule.py
# Khemchandani et al., DeepGraphMolGen [...]. J. Cheminform 2020
def reward_penalized_log_p(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    cycle_score = num_long_cycles(mol)

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def construct_problem(run_config):
    # We have to delay all importing of tensorflow until the child processes launch,
    # see https://github.com/tensorflow/tensorflow/issues/8220. We should be more careful about where / when we
    # import tensorflow, especially if there's a chance we'll use tf.serving to do the policy / reward evaluations on
    # the workers. Might require upstream changes to nfp as well.
    from rlmolecule.tree_search.reward import RankedRewardFactory
    from rlmolecule.tree_search.reward import LinearBoundedRewardFactory
    from rlmolecule.molecule.molecule_problem import MoleculeTFAlphaZeroProblem
    from rlmolecule.molecule.molecule_state import MoleculeState
    from rlmolecule.molecule.builder.builder import MoleculeBuilder

    class PenLogPOptimizationProblem(MoleculeTFAlphaZeroProblem):
        def get_initial_state(self) -> MoleculeState:
            return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self._config)

        def get_reward(self, state: MoleculeState) -> (float, {}):
            if state.forced_terminal:
                return reward_penalized_log_p(state.molecule), {'forced_terminal': True, 'smiles': state.smiles}
            return 0.0, {'forced_terminal': False, 'smiles': state.smiles}

    prob_config = run_config.problem_config
    builder = MoleculeBuilder(
        max_atoms=prob_config.get('max_atoms', 25),
        min_atoms=prob_config.get('min_atoms', 1),
        try_embedding=prob_config.get('tryEmbedding', True),
        sa_score_threshold=prob_config.get('sa_score_threshold', 4),
        stereoisomers=prob_config.get('stereoisomers', False),
        atom_additions=prob_config.get('atom_additions', ('C', 'N', 'O'))
    )

    engine = run_config.start_engine()

    run_id = run_config.run_id

    train_config = run_config.train_config
    if train_config.get('linear_reward'):
        reward_factory = LinearBoundedRewardFactory(min_reward=train_config.get('min_reward', 0),
                                                    max_reward=train_config.get('max_reward', 20))
    else:
        reward_factory = RankedRewardFactory(
            engine=engine,
            run_id=run_id,
            reward_buffer_min_size=train_config.get('reward_buffer_min_size', 10),
            reward_buffer_max_size=train_config.get('reward_buffer_max_size', 50),
            ranked_reward_alpha=train_config.get('ranked_reward_alpha', 0.75)
        )

    problem = PenLogPOptimizationProblem(
        engine,
        builder,
        run_id=run_id,
        reward_class=reward_factory,
        num_messages=train_config.get('num_messages', 1),
        num_heads=train_config.get('num_heads', 2),
        features=train_config.get('features', 8),
        max_buffer_size=train_config.get('max_buffer_size', 200),
        min_buffer_size=train_config.get('min_buffer_size', 15),
        batch_size=train_config.get('batch_size', 32),
        policy_checkpoint_dir=train_config.get(
            'policy_checkpoint_dir', 'policy_checkpoints')
    )

    return problem


def run_games(run_config):
    from rlmolecule.alphazero.alphazero import AlphaZero
    config = run_config.mcts_config
    game = AlphaZero(
        construct_problem(run_config),
        min_reward=config.get('min_reward', 0.0),
        pb_c_base=config.get('pb_c_base', 1.0),
        pb_c_init=config.get('pb_c_init', 1.25),
        dirichlet_noise=config.get('dirichlet_noise', True),
        dirichlet_alpha=config.get('dirichlet_alpha', 1.0),
        dirichlet_x=config.get('dirichlet_x', 0.25),
        # MCTS parameters
        ucb_constant=config.get('ucb_constant', math.sqrt(2)),
    )
    while True:
        path, reward = game.run(
            num_mcts_samples=config.get('num_mcts_samples', 50),
            max_depth=config.get('max_depth', 1000000),
        )
        logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1][0]}')


def train_model(run_config):
    config = run_config.train_config
    construct_problem(run_config).train_policy_model(
        steps_per_epoch=config.get('steps_per_epoch', 100),
        lr=float(config.get('lr', 1E-3)),
        epochs=int(float(config.get('epochs', 1E4))),
        game_count_delay=config.get('game_count_delay', 20),
        verbose=config.get('verbose', 2)
    )


def monitor(run_config):
    from rlmolecule.sql.tables import RewardStore
    problem = construct_problem(run_config)

    while True:
        best_reward = problem.session.query(RewardStore) \
            .filter_by(run_id=problem.run_id) \
            .order_by(RewardStore.reward.desc()).first()

        num_games = len(list(problem.iter_recent_games()))

        if best_reward:
            print(f"Best Reward: {best_reward.reward:.3f} for molecule "
                  f"{best_reward.data['smiles']} with {num_games} games played")

        time.sleep(5)


def setup_argparser():
    parser = argparse.ArgumentParser(
        description='Run the Penalized LogP optimization. Default is to run the script locally')

    parser.add_argument('--config', type=str,
                        help='Configuration file')
    parser.add_argument('--train-policy', action="store_true", default=False,
                        help='Train the policy model only (on GPUs)')
    parser.add_argument('--rollout', action="store_true", default=False,
                        help='Run the game simulations only (on CPUs)')

    return parser


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()

    run_config = RunConfig(args.config)

    if args.train_policy:
        train_model(run_config)
    elif args.rollout:
        # make sure the rollouts do not use the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        run_games(run_config)
    else:
        jobs = [multiprocessing.Process(target=monitor, args=(run_config,))]
        jobs[0].start()
        time.sleep(1)

        for i in range(5):
            jobs += [multiprocessing.Process(target=run_games, args=(run_config,))]

        jobs += [multiprocessing.Process(target=train_model, args=(run_config,))]

        for job in jobs[1:]:
            job.start()

        for job in jobs:
            job.join(300)
