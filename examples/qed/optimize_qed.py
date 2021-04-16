""" Optimize the Quantitative Estimate of Drug-likeness (QED)
See https://www.rdkit.org/docs/source/rdkit.Chem.QED.html
Starting point: a single carbon (C)
  - actions: add a bond or an atom
  - state: molecule state
  - reward: 0, unless a terminal state is reached, then the qed estimate of the molecule
"""

import argparse
import logging
import math
import multiprocessing
import os
import time

import rdkit
from rdkit.Chem.QED import qed

from examples.run_config import RunConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_problem(run_config):
    # We have to delay all importing of tensorflow until the child processes launch,
    # see https://github.com/tensorflow/tensorflow/issues/8220. We should be more careful about where / when we
    # import tensorflow, especially if there's a chance we'll use tf.serving to do the policy / reward evaluations on
    # the workers. Might require upstream changes to nfp as well.
    from rlmolecule.tree_search.reward import RankedRewardFactory
    from rlmolecule.molecule.molecule_problem import MoleculeTFAlphaZeroProblem
    from rlmolecule.molecule.molecule_state import MoleculeState
    from rlmolecule.molecule.builder.builder import MoleculeBuilder

    class QEDOptimizationProblem(MoleculeTFAlphaZeroProblem):
        def get_initial_state(self) -> MoleculeState:
            return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self._config)

        def get_reward(self, state: MoleculeState) -> (float, {}):
            if state.forced_terminal:
                return qed(state.molecule), {'forced_terminal': True, 'smiles': state.smiles}
            return 0.0, {'forced_terminal': False, 'smiles': state.smiles}

    prob_config = run_config.problem_config
    builder = MoleculeBuilder(
        max_atoms=prob_config.get('max_atoms', 25),
        min_atoms=prob_config.get('min_atoms', 1),
        tryEmbedding=prob_config.get('tryEmbedding', True),
        sa_score_threshold=prob_config.get('sa_score_threshold', 4),
        stereoisomers=prob_config.get('stereoisomers', False)
    )

    engine = run_config.start_engine()
    # engine = create_engine(f'sqlite:///qed_data.db',
    #                       connect_args={'check_same_thread': False},
    #                       execution_options = {"isolation_level": "AUTOCOMMIT"})

    run_id = run_config.run_id

    train_config = run_config.alphazero_config
    reward_factory = RankedRewardFactory(
        engine=engine,
        run_id=run_id,
        reward_buffer_min_size=train_config.get('reward_buffer_min_size', 10),
        reward_buffer_max_size=train_config.get('reward_buffer_max_size', 50),
        ranked_reward_alpha=train_config.get('ranked_reward_alpha', 0.75)
    )

    problem = QEDOptimizationProblem(
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
            'policy_checkpoints_dir', 'policy_checkpoints')
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
        epochs=int(config.get('epochs', 1E4)),
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
        description='Run the QED optimization. Default is to run the script locally')

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
