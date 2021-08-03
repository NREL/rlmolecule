import argparse
import logging
import math
import os
import pathlib

from rlmolecule.sql.run_config import RunConfig
from rlmolecule.tree_search.metrics import collect_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from stable_radical_problem import construct_problem


def run_games(run_config: RunConfig, **kwargs) -> None:
    from rlmolecule.alphazero.alphazero import AlphaZero
    config = run_config.mcts_config
    game = AlphaZero(
        construct_problem(run_config, **kwargs),
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
            timeout=config.get('timeout', None),
            max_depth=config.get('max_depth', 1000000),
        )
        logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1][0]}')


def train_model(run_config: RunConfig, **kwargs) -> None:
    config = run_config.train_config
    construct_problem(run_config, **kwargs).train_policy_model(
        steps_per_epoch=config.get('steps_per_epoch', 100),
        lr=float(config.get('lr', 1E-3)),
        epochs=int(float(config.get('epochs', 1E4))),
        game_count_delay=config.get('game_count_delay', 20),
        verbose=config.get('verbose', 2)
    )


def setup_argparser():
    parser = argparse.ArgumentParser(
        description='Optimize stable radicals to work as both the anode and cathode of a redox-flow battery.')

    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--train-policy',
                        action="store_true",
                        default=False,
                        help='Train the policy model only (on GPUs)')
    parser.add_argument('--rollout',
                        action="store_true",
                        default=False,
                        help='Run the game simulations only (on CPUs)')
    # '/projects/rlmolecule/pstjohn/models/20210214_radical_stability_new_data/',
    parser.add_argument('--stability-model',
                        '-S',
                        type=pathlib.Path,
                        required=True,
                        help='Radical stability model for computing the electron spin and buried volume')
    # '/projects/rlmolecule/pstjohn/models/20210214_redox_new_data/',
    parser.add_argument('--redox-model',
                        '-R',
                        type=pathlib.Path,
                        required=True,
                        help='Redox model for computing the ionization_energy and electron_affinity')
    # '/projects/rlmolecule/pstjohn/models/20210216_bde_new_nfp/',
    parser.add_argument('--bde-model',
                        '-B',
                        type=pathlib.Path,
                        required=True,
                        help='BDE model for computing the Bond Dissociation Energy')

    return parser


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()
    kwargs = vars(args)

    run_config = RunConfig(args.config)

    if args.train_policy:
        train_model(run_config, **kwargs)
    elif args.rollout:
        # make sure the rollouts do not use the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        run_games(run_config, **kwargs)
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
