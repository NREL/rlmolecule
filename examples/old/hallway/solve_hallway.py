import logging
import time
from typing import Tuple

import tensorflow as tf

import sqlalchemy
from sqlalchemy import create_engine

from rlmolecule.tree_search.reward import LinearBoundedRewardFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_problem(ranked_reward=True):
    
    from rlmolecule.tree_search.reward import RankedRewardFactory
    from rlmolecule.alphazero.tfalphazero_problem import TFAlphaZeroProblem

    from tf_model import policy_model  #todo: this looks broken? (psj)
    from hallway_config import HallwayConfig
    from hallway_state import HallwayState


    class HallwayProblem(TFAlphaZeroProblem):
        def __init__(self,
                     engine: sqlalchemy.engine.Engine,
                     model: tf.keras.Model,
                     config: HallwayConfig,
                     **kwargs) -> None:
            super(HallwayProblem, self).__init__(engine, model, **kwargs)
            self._config = config

        def get_initial_state(self) -> HallwayState:
            return HallwayState(1, 0, self._config)

        def get_reward(self, state: HallwayState) -> Tuple[float, dict]:
            reward = -1.0 * (state.steps + (self._config.size - state.position))
            return reward, {'position': state.position}

        def get_policy_inputs(self, state: HallwayState) -> dict:
            return {"position": [state.position], "steps": [state.steps]}

    config = HallwayConfig(size=16, max_steps=16)

    engine = create_engine(f'sqlite:///hallway_data.db',
                           connect_args={'check_same_thread': False},
                           execution_options = {"isolation_level": "AUTOCOMMIT"})

    model = policy_model(hidden_layers=1, hidden_dim=16)

    run_id = "hallway_example"

    if ranked_reward:
        reward_factory = RankedRewardFactory(
            engine=engine,
            run_id=run_id,
            reward_buffer_min_size=10,
            reward_buffer_max_size=50,
            ranked_reward_alpha=0.75
        )
    else:
        reward_factory = LinearBoundedRewardFactory(min_reward=-30, max_reward=-15)

    problem = HallwayProblem(
        engine,
        model,
        config,
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

    rewards_file = "_rewards.csv"
    #with open(rewards_file, "w") as f:  pass
    while True:
        path, reward = game.run(
            num_mcts_samples=num_mcts_samples,
            action_selection_function=MCTS.visit_selection if not use_az else None)

        print(path)

        print("REWARD:", reward.__dict__)
        if use_az:
            # Breaks if you use MCTS:
            logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1][0]}')
        #with open(rewards_file, "a") as f: 
        #    f.write(str(reward.raw_reward) + "\n")

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

    if 0:
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

    else:

        run_games(use_az=True, num_mcts_samples=64)

