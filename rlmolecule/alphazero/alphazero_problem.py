import os
import random
from abc import abstractmethod
from typing import Optional

import sqlalchemy

# from rlmolecule.alphazero.alphazero import AlphaZero
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.sql import Base, Session
from rlmolecule.sql.tables import Game, Reward
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class AlphaZeroProblem(MCTSProblem):

    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 run_id: Optional[str] = None,
                 max_buffer_size: int = 200,
                 batch_size: int = 32,
                 **kwargs):
        super(AlphaZeroProblem, self).__init__(**kwargs)
        Base.metadata.create_all(engine)
        Session.configure(bind=engine)
        self._session = Session()
        self.run_id = run_id if run_id is not None else os.environ.get('AZ_RUNID', 'not specified')
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size

    @property
    def session(self) -> 'sqlalchemy.orm.session.Session':
        return self._session

    def reward_wrapper(self, state: GraphSearchState) -> float:
        """A wrapper that caches reward calculations in a SQL database, and calls self.get_scaled_reward

        :param state: The state for which rewards are cached
        :return: the scaled reward
        """
        existing_record = self.session.query(Reward).get((hash(state), state.serialize()))
        if existing_record is not None:
            reward = existing_record.reward

        else:
            reward, data = self.get_reward(state)
            record = Reward(hash=hash(state),
                            state=state.serialize(),
                            run_id=self.run_id,
                            reward=reward,
                            data=data)
            self.session.add(record)
            self.session.commit()

        return self.get_scaled_reward(reward)

    def _store_search_statistics(self, path: [], reward: float, game: 'AlphaZero') -> None:
        """Store the game data in the replay buffer

        :param path: The path data collected by AlphaZero._accumulate_path_data
        :param reward: The final state's unscaled reward
        :param game: The game object just played
        """

        # path[-1] is the terminal state with no children
        search_statistics = [
            (vertex.state.serialize(), visit_probabilities)
            for (vertex, visit_probabilities) in path[:-1]]

        record = Game(id=str(self.id),
                      run_id=self.run_id,
                      raw_reward=reward,
                      scaled_reward=self.get_scaled_reward(reward),
                      search_statistics=search_statistics)

        self.session.add(record)
        self.session.commit()

    def iter_recent_games(self) -> (str, 'np.ndarray', float):
        """Iterate over randomly chosen positions in games from the replay buffer

        :returns: a generator of (serialized_parent, visit_probabilities, scaled_reward) pairs
        """

        recent_games = self.session.query(Game).filter_by(run_id=self.run_id) \
            .order_by(Game.index.desc()).limit(self.max_buffer_size)

        for game in recent_games:
            parent_state_string, visit_probabilities = random.choice(game.search_statistics)
            yield parent_state_string, game.scaled_reward, visit_probabilities

    def get_scaled_reward(self, reward: float) -> float:
        """
        Provide the ability to scale the reward returned by `get_reward`, i.e. in a RankedRewards scheme.
        Defaults to returning the unscaled reward.

        :param reward: the unscaled reward
        :return: The scaled reward
        """
        return reward

    @abstractmethod
    def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
        """
        A user-provided function to get value and child prior estimates for the given vertex.

        :return: (value_of_current_vertex, {child_vertex: child_prior for child_vertex in children})
        """
        pass
