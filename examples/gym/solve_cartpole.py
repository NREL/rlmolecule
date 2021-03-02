import logging
import time

import numpy as np
from sqlalchemy import create_engine

import gym

from tf_model import policy_model
from gym_problem import GymEnvProblem
from gym_state import GymEnvState
from alphazero_gym import AlphaZeroGymEnv


#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NOTE: These class definitions need to stay outside of construct_problem
# or you will error out on not being able to pickle/serialize them.

class CartPoleEnv(AlphaZeroGymEnv):
    """Lightweight wrapper around the gym env that makes the user implement
    the get_obs method."""

    def __init__(self, **kwargs):
        super().__init__(gym.envs.make("CartPole-v0"), **kwargs)
    
    def get_obs(self) -> np.ndarray:
        return np.array(self.state)


class CartPoleProblem(GymEnvProblem):
    """Cartpole TF AZ problem.  For now we will ask the user to implement
    any obs preprocessing directly in the get_policy_inputs method."""

    def __init__(self, 
                 engine: "sqlalchemy.engine.Engine",
                 **kwargs) -> None:
        env = CartPoleEnv()
        super().__init__(engine, env, **kwargs)

    def policy_model(self) -> "tf.keras.Model":
        return policy_model(obs_dim = self._env.observation_space.shape[0],
                            hidden_layers = 3,
                            hidden_dim = 16,)

    def get_policy_inputs(self, state: GymEnvState) -> dict:
        return {"obs": self._env.get_obs()}


def construct_problem():

    from rlmolecule.tree_search.reward import RankedRewardFactory

    engine = create_engine(f'sqlite:///cartpole_data.db',
                           connect_args={'check_same_thread': False},
                           execution_options = {"isolation_level": "AUTOCOMMIT"})

    run_id = "cartpole_example"

    reward_factory = RankedRewardFactory(
            engine=engine,
            run_id=run_id,
            reward_buffer_min_size=10,
            reward_buffer_max_size=50,
            ranked_reward_alpha=0.75
    )

    problem = CartPoleProblem(
        engine,
        run_id=run_id,
        reward_class=reward_factory,
        min_buffer_size=15,
        policy_checkpoint_dir='policy_checkpoints'
    )

    return problem


def run_games(use_az=True, num_mcts_samples=50):

    if use_az:
        from rlmolecule.alphazero.alphazero import AlphaZero
        game = AlphaZero(construct_problem(), dirichlet_noise=False)
    else:
        from rlmolecule.mcts.mcts import MCTS
        game = MCTS(construct_problem(ranked_reward=False))

    while True:
        path, reward = game.run(num_mcts_samples=num_mcts_samples)

        print(path)

        print("REWARD:", reward.__dict__)
        if use_az:
            logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1][0]}')
        

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


if __name__ == "__main__":

    import multiprocessing

    jobs = [multiprocessing.Process(target=monitor)]
    jobs[0].start()
    time.sleep(1)

    for i in range(5):
        jobs += [multiprocessing.Process(target=run_games)]

    jobs += [multiprocessing.Process(target=train_model)]

    for job in jobs[1:]:
        job.start()

    for job in jobs:
        job.join(300)

