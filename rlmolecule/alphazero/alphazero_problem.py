import logging
import os
import random
from abc import abstractmethod
from typing import List, Optional

import numpy as np
import sqlalchemy
from sqlalchemy import exc

from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.sql import Base, Session, digest, load_numpy_dict, serialize_ordered_numpy_dict
from rlmolecule.sql.tables import GameStore, RewardStore, StateStore
from rlmolecule.tree_search.graph_search_state import GraphSearchState
from rlmolecule.tree_search.reward import Reward, RewardFactory

logger = logging.getLogger(__name__)


class AlphaZeroProblem(MCTSProblem):

    def __init__(self, *,
                 reward_class: RewardFactory,
                 engine: sqlalchemy.engine.Engine,
                 run_id: Optional[str] = None,
                 max_buffer_size: int = 200,
                 min_buffer_size: int = 50,
                 batch_size: int = 32):

        super(AlphaZeroProblem, self).__init__(reward_class=reward_class)
        Base.metadata.create_all(engine, checkfirst=True)
        Session.configure(bind=engine)
        self._session = Session()
        self.run_id = run_id if run_id is not None else os.environ.get('AZ_RUNID', 'not specified')
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size

    @abstractmethod
    def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
        """
        A user-provided function to get value and child prior estimates for the given vertex.

        :return: (value_of_current_vertex, {child_vertex: child_prior for child_vertex in children})
        """
        pass

    @abstractmethod
    def get_policy_inputs(self, state: GraphSearchState) -> {str: np.ndarray}:
        """
        A user-provided function to calculate inputs for a given search state for the policy model. These are cached
        in StateTable for faster access during model training.

        todo: wrap this, make it a function of the vertex, and cache the results.

        :param state: A graph search state for which policy inputs are calculated
        :return: A dictionary of string keys and numpy array values
        """
        pass

    @property
    def session(self) -> 'sqlalchemy.orm.session.Session':
        return self._session

    def reward_wrapper(self, vertex: AlphaZeroVertex) -> Reward:
        """A wrapper that caches reward calculations in a SQL database, and calls self.get_scaled_reward

        :param vertex: The vertex for which state rewards are cached
        :return: the scaled reward
        """

        policy_digest, policy_inputs = self._get_policy_inputs_and_digest(vertex)

        existing_record = self.session.query(RewardStore).get((policy_digest, vertex.state.hash(), self.run_id))

        if existing_record is not None:
            reward = existing_record.reward

        else:
            reward, data = self.get_reward(vertex.state)
            record = RewardStore(digest=policy_digest,
                                 hash=vertex.state.hash(),
                                 run_id=self.run_id,
                                 reward=reward,
                                 data=data)

            try:
                # Here we handle the possibility of a race condition where another thread has already added the
                # current state's reward to the database before this thread has completed.
                self.session.merge(record)
                self.session.commit()
            except exc.IntegrityError:
                logger.debug(f"Duplicate reward entry encountered with {vertex.state}")
                self.session.rollback()

        return self.reward_class(raw_reward=reward)

    def policy_input_wrapper(self, vertex: AlphaZeroVertex) -> {str: np.ndarray}:
        """ Cache policy inputs in the vertex class

        :param vertex: A vertex for which policy inputs are desired
        :return: The policy input dictionary for the captive state
        """
        if getattr(vertex, 'policy_inputs', None) is None:
            vertex.policy_inputs = self.get_policy_inputs(vertex.state)

        return vertex.policy_inputs

    def _get_policy_inputs_and_digest(self, vertex: AlphaZeroVertex) -> (str, {str: np.ndarray}):
        """ Compute the policy inputs and the policy inputs digest for a given state in order to query the Reward or
        State tables

        :param state: A GraphSearchState to compute policy inputs for
        :return: policy_inputs_digest, policy_inputs
        """

        # todo: make this a function of the vertex, and cache the policy_inputs and digest
        policy_inputs = self.policy_input_wrapper(vertex)
        if getattr(vertex, 'policy_digest', None) is None:
            vertex.policy_digest = digest(serialize_ordered_numpy_dict(policy_inputs))

        return vertex.policy_digest, policy_inputs

    def maybe_store_state(self, vertex: AlphaZeroVertex) -> str:
        """ Add the serialized state to the StateStore table if it doesn't exist

        :param vertex: The state to store in the StateStore database (if it doesn't already exist)
        :return: the hash (message digest) of the given state
        """

        policy_digest, policy_inputs = self._get_policy_inputs_and_digest(vertex)
        state_hash = vertex.state.hash()

        if not self.session.query(StateStore).get((policy_digest, state_hash, self.run_id)):
            record = StateStore(digest=policy_digest,
                                hash=state_hash,
                                run_id=self.run_id,
                                state=vertex.state.serialize(),
                                policy_inputs=serialize_ordered_numpy_dict(policy_inputs))
            try:
                self.session.add(record)
                self.session.commit()
            except exc.IntegrityError:
                logger.debug(f"Duplicate state entry encountered with {vertex.state}")
                self.session.rollback()

        return policy_digest

    def store_search_statistics(self, path: [], reward: Reward) -> None:
        """Store the game data in the replay buffer

        :param path: The path data collected by AlphaZero._accumulate_path_data
        :param reward: The final state's unscaled reward
        """

        # path[-1] is the terminal state with no children
        search_statistics = []
        for parent, child_visits in path[:-1]:
            search_statistics += [
                (self.maybe_store_state(parent),
                 {self.maybe_store_state(child): visit_probability
                  for child, visit_probability in child_visits})
            ]

        record = GameStore(id=str(self.id),
                           run_id=self.run_id,
                           raw_reward=reward.raw_reward,
                           scaled_reward=reward.scaled_reward,
                           search_statistics=search_statistics)

        self.session.add(record)
        self.session.commit()

    def iter_recent_games(self) -> (List[str], List[float]):
        """Iterate over randomly chosen positions in games from the replay buffer

        :returns: a generator of (serialized_parent, visit_probabilities, scaled_reward) pairs
        """

        recent_games = self.session.query(GameStore).filter_by(run_id=self.run_id) \
            .order_by(GameStore.time.desc()).limit(self.max_buffer_size)

        for game in recent_games:
            parent_state_string, visit_probabilities = random.choice(game.search_statistics)

            yield ([parent_state_string] + list(visit_probabilities.keys()),
                   [game.scaled_reward] + list(visit_probabilities.values()))

    def lookup_policy_inputs_from_digest(self, policy_digest: str) -> {str: np.ndarray}:

        state_row = self.session.query(StateStore).filter_by(
            digest=policy_digest).filter_by(run_id=self.run_id).first()
        if not state_row:
            raise RuntimeError(f"Could not find state matching digest {policy_digest}")
        return load_numpy_dict(state_row.policy_inputs)
