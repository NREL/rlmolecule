import argparse
import logging
import math
import multiprocessing
import os
import time

from rlmolecule.sql.run_config import RunConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run_games(run_config: RunConfig, **kwargs) -> None:
    from rlmolecule.alphazero.alphazero import AlphaZero

    from stable_radical_problem import construct_problem

    logger.info("starting run_games script")

    config = run_config.mcts_config
    game = AlphaZero(
        construct_problem(run_config, **kwargs),
        min_reward=config.get("min_reward", 0.0),
        pb_c_base=config.get("pb_c_base", 1.0),
        pb_c_init=config.get("pb_c_init", 1.25),
        dirichlet_noise=config.get("dirichlet_noise", True),
        dirichlet_alpha=config.get("dirichlet_alpha", 1.0),
        dirichlet_x=config.get("dirichlet_x", 0.25),
        # MCTS parameters
        ucb_constant=config.get("ucb_constant", math.sqrt(2)),
    )
    while True:
        path, reward = game.run(
            num_mcts_samples=config.get("num_mcts_samples", 50),
            timeout=config.get("timeout", None),
            max_depth=config.get("max_depth", 1000000),
        )
        logger.info(
            f"Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1][0]}"
        )


def train_model(run_config: RunConfig, **kwargs) -> None:
    from stable_radical_problem import construct_problem

    logger.info("starting train_model script")

    config = run_config.train_config
    construct_problem(run_config, **kwargs).train_policy_model(
        steps_per_epoch=config.get("steps_per_epoch", 100),
        lr=float(config.get("lr", 1e-3)),
        epochs=int(float(config.get("epochs", 1e4))),
        game_count_delay=config.get("game_count_delay", 20),
        verbose=config.get("verbose", 2),
    )


def monitor(run_config: RunConfig, **kwargs):
    from rlmolecule.sql.tables import GameStore, RewardStore

    from stable_radical_problem import construct_problem

    logger.info("starting monitor script")
    problem = construct_problem(run_config, **kwargs)

    while True:
        best_reward = (
            problem.session.query(RewardStore)
            .filter_by(run_id=problem.run_id)
            .order_by(RewardStore.reward.desc())
            .first()
        )

        num_games = (
            problem.session.query(GameStore).filter_by(run_id=problem.run_id).count()
        )

        if best_reward:
            logger.info(
                f"Best Reward: {best_reward.reward:.3f} for molecule "
                f"{best_reward.data['smiles']} with {num_games} games played"
            )

        else:
            logger.debug("Monitor script looping, no reward found")

        time.sleep(5)


def setup_argparser():
    parser = argparse.ArgumentParser(
        description="Optimize stable radicals to work as both the anode"
        " and cathode of a redox-flow battery."
    )

    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument(
        "--train-policy",
        action="store_true",
        default=False,
        help="Train the policy model only (on GPUs)",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="Run the game simulations only (on CPUs)",
    )

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
    # else:
    #     logger.warning("Must specify either --train-policy or --rollout")

    else:

        # run_games(run_config)
        # train_model(run_config)

        jobs = [
            multiprocessing.Process(target=monitor, args=(run_config,), kwargs=kwargs)
        ]
        jobs[0].daemon = True
        jobs[0].start()
        time.sleep(1)

        for i in range(34):
            jobs += [
                multiprocessing.Process(
                    target=run_games, args=(run_config,), kwargs=kwargs
                )
            ]

        jobs += [
            multiprocessing.Process(
                target=train_model, args=(run_config,), kwargs=kwargs
            )
        ]

        for job in jobs[1:]:
            job.daemon = True
            job.start()

        # Wait for 30 minutes, and then exit
        start_time = time.time()
        while True:
            time.sleep(30)
            if (time.time() - start_time) > 1800:
                break