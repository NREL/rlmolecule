import logging

import sqlalchemy

from rlmolecule.sql.tables import RewardStore
from rlmolecule.tree_search.graph_search_state import GraphSearchState

logger = logging.getLogger(__name__)


def get_existing_reward(session: sqlalchemy.orm.session.Session,
                        run_id: str,
                        state: GraphSearchState):

    query = session.query(RewardStore).filter(RewardStore.hash == hash(state)).filter(RewardStore.run_id == run_id)
    num_results = query.count()
    if num_results == 0:
        return None

    elif num_results == 1:
        return query.one()

    # This must be a hash collision
    else:
        logger.debug(f"Hash collision observed for state {state} with hash {hash(state)}")
        return query.filter(RewardStore.state == state.serialize()).one()


