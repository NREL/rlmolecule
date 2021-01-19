from typing import Optional, Sequence

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class HallwayState(GraphSearchState):
    def __init__(self, 
                 position: int,
                 config: any,
                 force_terminal: bool = False
                 ) -> None:

        self._config: any = config
        self._position: int = position
        self._forced_terminal: bool = force_terminal

    def __repr__(self) -> str:
        return f"{self._position}{' (t)' if self._forced_terminal else ''}"

    def equals(self, other: any) -> bool:
        return type(self) == type(other) and \
               self._position == other._position and \
               self._forced_terminal == other._forced_terminal 

    def hash(self) -> int:
        return hash(self.__repr__()) ^ (13 * self._forced_terminal)

    def get_next_actions(self) -> Sequence['HallwayState']:
        result = []
        if not self._forced_terminal:
            if self.position == 0 or self.position == self.config.size - 1:
                result.append(HallwayState(self.position, self.config, force_terminal=True))
            else:
                result.extend([HallwayState(self.position-1, self.config),
                               HallwayState(self.position+1, self.config)])        
        return result

    @property
    def forced_terminal(self) -> bool:
        return self._forced_terminal

    @property
    def config(self) -> any:
        return self._config

    @property
    def position(self) -> int:
        return self._position
