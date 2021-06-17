import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import sqlalchemy

from rlmolecule.sql import Session
from rlmolecule.sql.tables import GameStore
from rlmolecule.tree_search.metrics import collect_metrics

logger = logging.getLogger(__name__)


class Reward:
    def __init__(self, *, raw_reward: Optional[float] = None, scaled_reward: Optional[float] = None):
        """ Class to coordinate between rewards that are raw (provided by the user) and scaled (between zero and one
        using some type of reward scaling)

        :param raw_reward: Reward as calculated by the defined `get_reward` function.
        :param scaled_reward: Rewards after applying some type of scaling function
        """
        self._raw_reward = raw_reward
        self._scaled_reward = scaled_reward

        if scaled_reward is not None:
            assert 0 <= scaled_reward <= 1, "scaled reward should likely be between zero and one, was this intended?"

    @property
    def raw_reward(self) -> float:
        if self._raw_reward is not None:
            return self._raw_reward
        else:
            raise RuntimeError("Attempting to access non-provided raw reward")

    @property
    def scaled_reward(self) -> float:
        if self._scaled_reward is not None:
            return self._scaled_reward
        else:
            raise RuntimeError("Attempting to access non-provided scaled reward")


class RewardFactory(ABC):
    @abstractmethod
    def _scale(self, reward: float) -> float:
        pass

    def initialize_run(self) -> None:
        """Any pre-run initialization that might be required"""
        pass

    @collect_metrics
    def __call__(self, *, raw_reward: Optional[float] = None, scaled_reward: Optional[float] = None) -> Reward:
        """ Initialize a Reward class with raw and scaled rewards, scaling the raw reward if a scaled reward is not
        provided. Allows a pass-through of a scaled reward if scaled_reward is provided

        :param raw_reward: A reward on the same scale as the user-defined reward function
        :param scaled_reward: A reward that has been pre-scaled between zero and one according to some scaling scheme
        :return: An initialized Reward class
        """
        if scaled_reward is None:
            scaled_reward = self._scale(raw_reward)

        return Reward(raw_reward=raw_reward, scaled_reward=scaled_reward)


class RawRewardFactory(RewardFactory):
    """Just passes the raw reward through as the scaled reward"""
    def _scale(self, reward: float) -> float:
        return reward


class LinearBoundedRewardFactory(RewardFactory):
    """Maps rewards to the 0->1 range, where the minimum reward maps to zero and the maximum reward maps to 1. Values
    above and below the max / min reward are capped at 0 or 1 respectively. """
    def __init__(self, min_reward: float = 0., max_reward: float = 1.) -> None:
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
    def __init__(
        self,
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
        self.r_alpha = False

    def initialize_run(self):
        """Query the game database to determine a suitable r_alpha threshold and store the threshold for later use.
        """
        all_games = self._session.query(GameStore).filter_by(run_id=self.run_id)

        n_games = all_games.count()
        if n_games >= self._reward_buffer_min_size:
            buffer_games = all_games.order_by(GameStore.time.desc()).limit(self._reward_buffer_max_size)
            buffer_raw_rewards = self._session.query(buffer_games.subquery().c.raw_reward).all()
            self.r_alpha = np.percentile(np.array(buffer_raw_rewards),
                                         100 * self._ranked_reward_alpha,
                                         interpolation='lower')
        else:
            logger.debug(f"ranked_reward: not enough games ({n_games})")
            self.r_alpha = None

    def _scale(self, reward: float) -> float:
        """Convert the continuous reward value into either a win or loss based on the previous games played.
        If the reward for this game is > the previous *ranked_reward_alpha* fraction of games (e.g., 90% of games), 
        then this reward is a win. Otherwise, it's a loss.
        """

        logger.debug(f"ranked_reward: r_alpha={self.r_alpha}, reward={reward}")

        assert self.r_alpha is not False, "ranked rewards not properly initialized"

        if self.r_alpha is None:
            # Here, we don't have enough of a game buffer
            # to decide if the move is good or not
            return np.random.choice([0., 1.])

        if np.isclose(reward, self.r_alpha):
            return np.random.choice([0., 1.])

        elif reward > self.r_alpha:
            return 1.

        elif reward < self.r_alpha:
            return 0.
