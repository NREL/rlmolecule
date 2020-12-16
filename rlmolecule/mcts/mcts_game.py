import math
import uuid


class MCTSGame:
    def __init__(self, ucb_constant: float = math.sqrt(2)) -> None:
        self.__id: uuid.UUID = uuid.uuid4()
        self.ucb_constant = ucb_constant

    @property
    def id(self) -> uuid.UUID:
        return self.__id