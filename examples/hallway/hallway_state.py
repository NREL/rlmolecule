from typing import Optional, Sequence

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class HallwayState(GraphSearchState):
    def __init__(self, 
                 position: int,
                 steps: int,
                 config: any,
                 ) -> None:

        self._position: int = position
        self._steps: int = steps
        self._config: any = config

    def __repr__(self) -> str:
        return f"{self._position},{self._steps}"

    def equals(self, other: any) -> bool:
        return type(self) == type(other) and \
               self._position == other._position and \
               self._steps == other._steps

    def hash(self) -> int:
        return hash(self.__repr__())

    def get_next_actions(self) -> Sequence['HallwayState']:
        if self.position == self.config.size or self.steps == self.config.max_steps:
            return []   # This is how you decalre "terminal"
        else:
            # Otherwise, return the new left/right positions.  We imagine there
            # is a wall at the leftmost endpoint, so going left from position 1 
            # has no effect on position.
            return [HallwayState(max(1, self.position-1), self.steps+1, self.config),
                    HallwayState(self.position+1, self.steps+1, self.config)]

    @property
    def config(self) -> any:
        return self._config

    @property
    def position(self) -> int:
        return self._position

    @property
    def steps(self) -> int:
        return self._steps
