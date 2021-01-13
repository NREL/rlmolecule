import logging

import numpy as np

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.sql.tables import Game

logger = logging.getLogger(__name__)

class RankedRewards(AlphaZeroProblem):
    def __init__(self, *args,
                 reward_buffer_min_size:int = 50,
                 reward_buffer_max_size: int = 250,
                 ranked_reward_alpha: float = 0.9,
                 **kwargs):

        super(RankedRewards, self).__init__(*args, **kwargs)
        self._reward_buffer_min_size = reward_buffer_min_size
        self._reward_buffer_max_size = reward_buffer_max_size
        self._ranked_reward_alpha = ranked_reward_alpha


    def get_scaled_reward(self, reward: float) -> float:

        all_games = self.session.query(Game).filter_by(run_id=self.run_id)
        n_games = all_games.count()

        if n_games < self._reward_buffer_min_size:
            # Here, we don't have enough of a game buffer
            # to decide if the move is good or not
            logger.debug(f"ranked_reward: not enough games ({n_games})")
            return np.random.choice([0., 1.])

        all_games.order_by(Game.index.desc()).limit(self._reward_buffer_max_size)
        #todo: figure out percentile_disc with declarative base interface


