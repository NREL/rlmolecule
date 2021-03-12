import logging
import time
import sys
from typing import Tuple

import numpy as np
from sqlalchemy import create_engine

import gym

from examples.gym.tf_model import policy_model_2
from examples.gym.gym_problem import GymEnvProblem
from examples.gym.gym_state import GymEnvState
from examples.gym.alphazero_gym import AlphaZeroGymEnv
import gridworld_env as gw

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# NOTE: These class definitions need to stay outside of construct_problem
# or you will error out on not being able to pickle/serialize them.
class GridWorldEnv(AlphaZeroGymEnv, gym.ObservationWrapper):
    def __init__(self,
                 grid: np.ndarray,
                 max_episode_steps: int = 20,
                 goal_reward: float = 100.,
                 **kwargs):
        env = gw.GridEnv(grid, max_episode_steps=max_episode_steps, goal_reward=goal_reward)
        super().__init__(env, **kwargs)

    def reset(self):
        return self.env.reset()

    def observation(self, obs) -> np.ndarray:
        return obs / 255.

    def get_obs(self) -> np.ndarray:
        return self.observation(self.env.get_obs())


class GridWorldProblem(GymEnvProblem):
    def __init__(self, 
                 grid: np.ndarray,
                 engine: "sqlalchemy.engine.Engine",
                 **kwargs) -> None:
        env = GridWorldEnv(grid)
        super().__init__(engine, env, **kwargs)

    def policy_model(self) -> "tf.keras.Model":
        obs_shape = self.env.reset().shape
        return policy_model_2(obs_dim=obs_shape,
                              hidden_layers=1,
                              conv_layers=1,
                              filters_dim=[16],
                              kernel_dim=[2],
                              strides_dim=[1],
                              hidden_dim=64)

    def get_policy_inputs(self, state: GymEnvState) -> dict:
        return {"obs": np.expand_dims(state.env.get_obs(), axis=-1)}

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
            reward_buffer_min_size=10,
            reward_buffer_max_size=50,
            ranked_reward_alpha=0.75
    )

    grid = np.zeros((3, 4, 4), dtype=int)
    grid[gw.PLAYER_CHANNEL, 0, 0] = 1
    grid[gw.GOAL_CHANNEL, -1, -1] = 1

    problem = GridWorldProblem(
        grid,
        engine,
        run_id=run_id,
        reward_class=reward_factory,
        min_buffer_size=15,
        policy_checkpoint_dir='gridworld_policy_checkpoints'
    )

    return problem


def run_games(use_az=False, num_mcts_samples=50, num_games=None):

    if use_az:
        from rlmolecule.alphazero.alphazero import AlphaZero
        game = AlphaZero(construct_problem(), dirichlet_noise=False)
    else:
        from rlmolecule.mcts.mcts import MCTS
        game = MCTS(construct_problem())

    # TODO: here, allow max games
    num_games = num_games if num_games is not None else sys.maxsize
    for _ in range(num_games):
        start_time = time.time()
        path, reward = game.run(num_mcts_samples=num_mcts_samples)
        elapsed = time.time() - start_time
        print(path)

        print("REWARD:", reward.__dict__)
        if use_az:
            logger.info((f'Game Finished -- Reward {reward.raw_reward:.3f}' \
                          ' -- Final state {path[-1][0]}' \
                          ' -- CPU time {elapsed} (s)'))
        

def train_model():
    construct_problem().train_policy_model(steps_per_epoch=100,
                                           game_count_delay=20,
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

    args = parser.parse_args()

    if args.train_policy:
        train_model()
    elif args.rollout:
        run_games(num_mcts_samples=args.num_mcts_samples, num_games=args.num_games)
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

