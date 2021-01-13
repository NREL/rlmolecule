import multiprocessing
import logging

import rdkit
from rdkit.Chem.QED import qed
from sqlalchemy import create_engine

from rlmolecule.alphazero.alphazero import AlphaZero
from rlmolecule.alphazero.reward import RankedRewardFactory
from rlmolecule.molecule.molecule_config import MoleculeConfig
from rlmolecule.molecule.molecule_problem import MoleculeAlphaZeroProblem
from rlmolecule.molecule.molecule_state import MoleculeState
from rlmolecule.sql.tables import RewardStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QEDOptimizationProblem(MoleculeAlphaZeroProblem):

    def __init__(self,
                 engine: 'sqlalchemy.engine.Engine',
                 config: 'MoleculeConfig', **kwargs) -> None:
        super(QEDOptimizationProblem, self).__init__(engine, **kwargs)
        self._config = config

    def get_initial_state(self) -> MoleculeState:
        return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self._config)

    def get_reward(self, state: MoleculeState) -> (float, {}):
        if state.forced_terminal:
            return qed(state.molecule), {'forced_terminal': True}
        return 0.0, {'forced_terminal': False}

config = MoleculeConfig(max_atoms=4,
                        min_atoms=1,
                        tryEmbedding=False,
                        sa_score_threshold=None,
                        stereoisomers=False)

engine = create_engine(f'sqlite:///qed_data.db',
                       connect_args={'check_same_thread': False})

run_id = 'qed_example'

reward_factory = RankedRewardFactory(
    engine=engine,
    run_id=run_id,
    reward_buffer_min_size=10,
    reward_buffer_max_size=50,
    ranked_reward_alpha=0.75
)

problem = QEDOptimizationProblem(
    engine,
    config,
    run_id=run_id,
    reward_class=reward_factory,
    features=8,
    num_heads=2,
    num_messages=1,
    policy_checkpoint_dir='policy_checkpoints')

def run_games():
    game = AlphaZero(problem)
    while True:
        path, reward = game.run(num_mcts_samples=50)
        logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1][0]}')


def train_model():
    problem.train_policy_model(steps_per_epoch=100)

# def monitor():
#
#     while True:
#         best_reward = problem.session.query(RewardStore) \
#             .filter_by(run_id=problem.run_id) \
#             .order_by(RewardStore.reward.desc()).first().reward
#
#         print(f"Best Reward: {best_reward}")
#
#         time.sleep(1)


if __name__ == "__main__":
    run_games()
    #
    # job = multiprocessing.Process(target=run_games)
    # job.start()
    # job.join(300)

    #
    # p = multiprocessing.Process(target=train_model)
    # p.start()

    # p = multiprocessing.Process(target=monitor)
    # p.start()
    # p.join(timeout=300)
