import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import sqlalchemy

from rlmolecule.sql import Session
from rlmolecule.sql.tables import GameStore

logger = logging.getLogger(__name__)


class Reward:
    def __init__(self, raw_reward: float, scaled_reward: float):
        self.raw_reward = raw_reward
        self.scaled_reward = scaled_reward


class RewardFactory(ABC):
    @abstractmethod
    def _scale(self, reward: float) -> float:
        pass

    def __call__(self, reward):
        return Reward(reward, self._scale(reward))


class RawRewardFactory(RewardFactory):
    """Just passes the raw reward through as the scaled reward"""
    def _scale(self, reward: float) -> float:
        return reward


class LinearBoundedRewardFactory(RewardFactory):
    """Maps rewards to the 0->1 range, where the minimum reward maps to zero and the maximum reward maps to 1. Values
    above and below the max / min reward are capped at 0 or 1 respectively. """
    def __init__(self,
                 min_reward: float = 0.,
                 max_reward: float = 1.) -> None:
        self.min_reward = min_reward
        self.max_reward = max_reward

    def _scale(self, reward: float) -> float:

        scaled_reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        return float(np.clip(scaled_reward, 0, 1))


# todo: not ideal to have run_id and engine as init args, but otherwise the rewardFactory and problem classes have a
#  circular dependency. Ultimately it would probably better to do all the `reward_wrapper` logic in these classes,
#  including caching results in a Reward buffer. But then we'd still need some way of syncing the run_id and engine
#  parameters between the reward and problem classes. I didn't want to combine these classes, since then it might be
#  tricky to modularize which reward scaling scheme we wanted to use.
class RankedRewardFactory(RewardFactory):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 run_id: Optional[str] = None,
                 reward_buffer_min_size: int = 50,
                 reward_buffer_max_size: int = 250,
                 ranked_reward_alpha: float = 0.90,
                 ) -> None:

        Session.configure(bind=engine)
        self._session = Session()
        self.run_id = run_id if run_id is not None else os.environ.get('AZ_RUNID', 'not specified')
        self._reward_buffer_min_size = reward_buffer_min_size
        self._reward_buffer_max_size = reward_buffer_max_size
        self._ranked_reward_alpha = ranked_reward_alpha
        self._has_enough_games = False

    def _scale(self, reward: float) -> float:

        all_games = self._session.query(GameStore).filter_by(run_id=self.run_id)

        if not self._has_enough_games:
            n_games = all_games.count()
            if n_games < self._reward_buffer_min_size:
                # Here, we don't have enough of a game buffer
                # to decide if the move is good or not
                logger.debug(f"ranked_reward: not enough games ({n_games})")
                return np.random.choice([0., 1.])
            else:
                self._has_enough_games = True  # todo: write tests?

        buffer_games = all_games.order_by(GameStore.index.desc()).limit(self._reward_buffer_max_size)
        buffer_raw_rewards = self._session.query(buffer_games.subquery().c.raw_reward).all()

        r_alpha = np.percentile(np.array(buffer_raw_rewards),
                                100 * self._ranked_reward_alpha,
                                interpolation='lower')

        logger.debug(f"ranked_reward: r_alpha={r_alpha}, reward={reward}")

        if np.isclose(reward, r_alpha):
            return np.random.choice([0., 1.])

        elif reward > r_alpha:
            return 1.

        elif reward < r_alpha:
            return 0.
