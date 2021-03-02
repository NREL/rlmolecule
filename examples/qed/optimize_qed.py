import logging
import multiprocessing
import time

import rdkit
from rdkit.Chem.QED import qed
from sqlalchemy import create_engine

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_problem():
    # We have to delay all importing of tensorflow until the child processes launch,
    # see https://github.com/tensorflow/tensorflow/issues/8220. We should be more careful about where / when we
    # import tensorflow, especially if there's a chance we'll use tf.serving to do the policy / reward evaluations on
    # the workers. Might require upstream changes to nfp as well.
    from rlmolecule.tree_search.reward import RankedRewardFactory
    from rlmolecule.molecule.molecule_config import MoleculeConfig
    from rlmolecule.molecule.molecule_problem import MoleculeTFAlphaZeroProblem
    from rlmolecule.molecule.molecule_state import MoleculeState

    class QEDOptimizationProblem(MoleculeTFAlphaZeroProblem):

        def __init__(self,
                     engine: 'sqlalchemy.engine.Engine',
                     config: 'MoleculeConfig', **kwargs) -> None:
            super(QEDOptimizationProblem, self).__init__(engine, config, **kwargs)
            self._config = config

        def get_initial_state(self) -> MoleculeState:
            return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self._config)

        def get_reward(self, state: MoleculeState) -> (float, {}):
            if state.forced_terminal:
                return qed(state.molecule), {'forced_terminal': True, 'smiles': state.smiles}
            return 0.0, {'forced_terminal': False, 'smiles': state.smiles}

    config = MoleculeConfig(max_atoms=25,
                            min_atoms=1,
                            tryEmbedding=True,
                            sa_score_threshold=4.,
                            stereoisomers=False)

    engine = create_engine(f'sqlite:///qed_data.db',
                           connect_args={'check_same_thread': False},
                           execution_options = {"isolation_level": "AUTOCOMMIT"})

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

        if best_reward:
            print(f"Best Reward: {best_reward.reward:.3f} for molecule "
                  f"{best_reward.data['smiles']} with {num_games} games played")

        time.sleep(5)


if __name__ == "__main__":

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
