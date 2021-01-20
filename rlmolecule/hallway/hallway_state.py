from typing import Optional, Sequence

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class HallwayState(GraphSearchState):
    def __init__(self, 
                 position: int,
                 steps: int,
                 config: any,
                 #terminal: bool = False
                 ) -> None:

        self._position: int = position
        self._steps: int = steps
        self._config: any = config
        #self._terminal: bool = terminal

    def __repr__(self) -> str:
        return f"{self._position},{self._steps}"

    def equals(self, other: any) -> bool:
        return type(self) == type(other) and \
               self._position == other._position and \
               self._steps == other._steps #and \
               #self._terminal == other._terminal 

    def hash(self) -> int:
        #return hash(self.__repr__()) ^ (13 * self._terminal)
        return hash(self.__repr__())

    def get_next_actions(self) -> Sequence['HallwayState']:
        # if self.terminal:
        #     return []
        # if self.position == 0 or self.position == self.config.size - 1 or self.steps == self.config.max_steps:
        #     return [HallwayState(self.position, self.steps, self.config, terminal=True)]
        # else:
        #     return [HallwayState(self.position-1, self.steps+1, self.config),
        #             HallwayState(self.position+1, self.steps+1, self.config)]
        if self.position == 0 or \
           self.position == self.config.size - 1 or \
           self.steps == self.config.max_steps:
            return []
        else:
            return [HallwayState(self.position-1, self.steps+1, self.config),
                    HallwayState(self.position+1, self.steps+1, self.config)]

    # @property
    # def terminal(self) -> bool:
    #     return self._terminal

    @property
    def config(self) -> any:
        return self._config

    @property
    def position(self) -> int:
        return self._position

    @property
    def steps(self) -> int:
        return self._steps
