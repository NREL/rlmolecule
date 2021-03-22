import logging
import time
import sys
from typing import Tuple

import numpy as np
from sqlalchemy import create_engine

from rlmolecule.gym.gym_problem import GymEnvProblem
from rlmolecule.gym.gym_state import GymEnvState
from rlmolecule.gym.alphazero_gym import AlphaZeroGymEnv

from examples.gym.tf_model import policy_model_2
import examples.gym.gridworld_env as gw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NOTE: These class definitions need to stay outside of construct_problem
# or you will error out on not being able to pickle/serialize them.
class GridWorldEnv(AlphaZeroGymEnv):
    def __init__(self,
                 configured_env: gw.GridEnv,
                 **kwargs):
        super().__init__(configured_env, **kwargs)

    def reset(self):
        return self.env.reset()

    def get_obs(self) -> np.ndarray:
        return self.env.get_obs()


class GridWorldProblem(GymEnvProblem):
    def __init__(self, 
                 env: gw.GridEnv,
                 engine: "sqlalchemy.engine.Engine",
                 **kwargs) -> None:
        super().__init__(engine, env, **kwargs)

    def policy_model(self) -> "tf.keras.Model":
        obs_shape = self.env.reset().shape
        return policy_model_2(obs_dim=obs_shape,
                              hidden_layers=2,
                              conv_layers=1,
                              filters_dim=[16],
                              kernel_dim=[2],
                              strides_dim=[1],
                              hidden_dim=64)

    def get_policy_inputs(self, state: GymEnvState) -> dict:
        return {
            "obs": state.env.get_obs(),
            "steps": 0.*np.array([np.float64(self.env.episode_steps)])
        }

    def get_reward(self, state: GymEnvState) -> Tuple[float, dict]:
        return state.cumulative_reward, {}


def construct_problem():

    from rlmolecule.tree_search.reward import RankedRewardFactory

    engine = create_engine(f'sqlite:///gridworld_data.db',
                           connect_args={'check_same_thread': False},
                           execution_options = {"isolation_level": "AUTOCOMMIT"})

    run_id = "gridworld_example"

    reward_factory = RankedRewardFactory(
            engine=engine,
            run_id=run_id,
            reward_buffer_min_size=20,
            reward_buffer_max_size=20,
            ranked_reward_alpha=0.75
    )

    grid = np.zeros((3, 5, 5), dtype=np.float64)
    grid[gw.PLAYER_CHANNEL, 0, 0] = 1
    grid[gw.GOAL_CHANNEL, -1, -1] = 1
    env = gw.GridEnv(grid, max_episode_steps=12)

    problem = GridWorldProblem(
        env,
        engine,
        run_id=run_id,
        reward_class=reward_factory,
        min_buffer_size=10,
        max_buffer_size=10,
        batch_size=32,
        policy_checkpoint_dir='gridworld_policy_checkpoints'
    )

    return problem


def run_games(use_mcts=False, num_mcts_samples=64, num_games=None):

    if use_mcts:
        from rlmolecule.mcts.mcts import MCTS
        game = MCTS(construct_problem())
    else:
        from rlmolecule.alphazero.alphazero import AlphaZero
        game = AlphaZero(construct_problem(), dirichlet_noise=False)

    # TODO: here, allow max games
    num_games = num_games if num_games is not None else sys.maxsize
    for _ in range(num_games):
        start_time = time.time()
        path, reward = game.run(num_mcts_samples=num_mcts_samples)
        elapsed = time.time() - start_time
        print("REWARD:", reward.__dict__)
        logger.info(('Game Finished -- Reward {:.3f}'.format(reward.raw_reward) +
                      ' -- Final state {}'.format(path[-1]) +
                      ' -- CPU time {:1.3f} (s)'.format(elapsed)))

def train_model():
    construct_problem().train_policy_model(steps_per_epoch=100,
                                           game_count_delay=10,
                                           verbose=2)


def monitor():

    from rlmolecule.sql.tables import RewardStore
    problem = construct_problem()

    while True:
        best_reward = problem.session.query(RewardStore) \
            .filter_by(run_id=problem.run_id) \
            .order_by(RewardStore.reward.desc()).first()

        num_games = len(list(problem.iter_recent_games()))

        if hasattr(best_reward, "data") and "position" in best_reward.data:
            print(f"Best Reward: {best_reward.reward:.3f} for final position "
                  f"{best_reward.data['position']} with {num_games} games played")

        time.sleep(5)


def setup_argparser():
    

    return parser


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Solve the Hallway problem (move from one side of the hallway to the other). ' +
                'Default is to run multiple games and training using multiprocessing')

    parser.add_argument('--train-policy', action="store_true", default=False,
                        help='Train the policy model only (on GPUs)')
    parser.add_argument('--rollout', action="store_true", default=False,
                        help='Run the game simulations only (on CPUs)')
    parser.add_argument('--num-workers', type=int, default=7,
                        help='Number of multiprocessing workers when running local rollout')
    parser.add_argument("--num-games", type=int, default=None)
    parser.add_argument("--num-mcts-samples", type=int, default=50)
    parser.add_argument("--use-mcts", action="store_true")
    parser.add_argument("--log-level", default="INFO", type=str)

    args = parser.parse_args()

    logger.setLevel(level=getattr(logging, args.log_level.upper()))

    if args.train_policy:
        train_model()
    elif args.rollout:
        run_games(
            use_mcts=args.use_mcts,
            num_mcts_samples=args.num_mcts_samples,
            num_games=args.num_games)
    else:
        assert args.num_workers >= 3  # need at least 3 workers here...

        import multiprocessing

        jobs = [multiprocessing.Process(target=monitor)]
        jobs[0].start()
        time.sleep(1)

        for i in range(args.num_workers - 2):
            jobs += [multiprocessing.Process(target=run_games)]

        jobs += [multiprocessing.Process(target=train_model)]

        for job in jobs[1:]:
            job.start()

        for job in jobs:
            job.join(300)

