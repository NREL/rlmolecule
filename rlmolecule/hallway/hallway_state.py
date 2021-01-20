from typing import Optional, Sequence

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class HallwayState(GraphSearchState):
    def __init__(self, 
                 position: int,
                 steps: int,
                 config: any,
                 terminal: bool = False
                 ) -> None:

        self._position: int = position
        self._steps: int = steps
        self._config: any = config
        self._terminal: bool = terminal

    def __repr__(self) -> str:
        return f"{self._position} [{self._steps}] {' (t)' if self._terminal else ''}"

    def equals(self, other: any) -> bool:
        return type(self) == type(other) and \
               self._position == other._position and \
               self._steps == other._steps and \
               self._terminal == other._terminal 

    def hash(self) -> int:
        return hash(self.__repr__()) ^ (13 * self._terminal)

    def get_next_actions(self) -> Sequence['HallwayState']:
        result = []
        if not self.terminal:
            if self.position == 0 or self.position == self.config.size or self.steps == self.config.max_steps:
                result.append(HallwayState(self.position, self.steps, self.config, terminal=True))
            else:
                steps = self.steps + 1
                result.extend([HallwayState(self.position-1, steps, self.config),
                               HallwayState(self.position+1, steps, self.config)])        
        return result

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def config(self) -> any:
        return self._config

    @property
    def position(self) -> int:
        return self._position

    @property
    def steps(self) -> int:
        return self._steps
