import logging
import multiprocessing
import time

from sqlalchemy import create_engine

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_problem():
    from rlmolecule.alphazero.reward import RankedRewardFactory
    from rlmolecule.hallway.hallway_config import HallwayConfig
    from rlmolecule.hallway.hallway_problem import HallwayAlphaZeroProblem
    from rlmolecule.hallway.hallway_state import HallwayState

    class HallwayProblem(HallwayAlphaZeroProblem):
        def __init__(self,
                     engine: 'sqlalchemy.engine.Engine',
                     config: 'HallwayConfig',
                     **kwargs) -> None:
            super(HallwayProblem, self).__init__(engine, **kwargs)
            self._config = config

        
        def get_initial_state(self) -> HallwayState:
            return HallwayState(int(self._config.size/2), self._config)

        def get_reward(self, state: HallwayState) -> (float, {}):
            if state.forced_terminal:
                return self._config.terminal_reward, {'forced_terminal': True, 'position': state.position}
            else:
                return self._config.step_reward, {'forced_terminal': False, 'position': state.position}

    config = HallwayConfig()

    engine = create_engine(f'sqlite:///hallway_data.db',
                           connect_args={'check_same_thread': False},
                           execution_options = {"isolation_level": "AUTOCOMMIT"})

    run_id = "hallway_example"

    reward_factory = RankedRewardFactory(
        engine=engine,
        run_id=run_id,
        reward_buffer_min_size=10,
        reward_buffer_max_size=50,
        ranked_reward_alpha=0.75
    )

    problem = HallwayProblem(
        engine,
        config,
        run_id=run_id,
        reward_class=reward_factory,
        hallway_size=config.size,
        features=8,
        hidden_layers=3,
        hidden_dim=16,
        min_buffer_size=15,
        policy_checkpoint_dir='policy_checkpoints'
    )

    return problem



def run_games():
    from rlmolecule.alphazero.alphazero import AlphaZero
    game = AlphaZero(construct_problem())
    while True:
        path, reward = game.run(num_mcts_samples=50)
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

        if "position" in best_reward.data:
            print(f"Best Reward: {best_reward.reward:.3f} for final position "
                  f"{best_reward.data['position']} with {num_games} games played")

        time.sleep(5)


if __name__ == "__main__":

    if 1:
        jobs = [multiprocessing.Process(target=monitor)]
        jobs[0].start()
        time.sleep(1)

        for i in range(2):
            jobs += [multiprocessing.Process(target=run_games)]

        jobs += [multiprocessing.Process(target=train_model)]

        for job in jobs[1:]:
            job.start()

        for job in jobs:
            job.join(300)

    else:

        run_games()