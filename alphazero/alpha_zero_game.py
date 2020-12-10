import logging
import uuid
from abc import abstractmethod

from alphazero.alphazero_node import AlphaZeroNode

logger = logging.getLogger(__name__)


class AlphaZeroGame:
    
    def __init__(self,
                 min_reward: float = 0.0,
                 pb_c_base: float = 1.0,
                 pb_c_init: float = 1.25,
                 dirichlet_noise: bool = True,
                 dirichlet_alpha: float = 1.0,
                 dirichlet_x: float = 0.25,
                 ) -> None:
        """
        Constructor.
        :param min_reward: Minimum reward to return for invalid actions
        :param pb_c_base: 19652 in pseudocode
        :param pb_c_init:
        :param dirichlet_noise: whether to add dirichlet noise
        :param dirichlet_alpha: dirichlet 'shape' parameter. Larger values spread out probability over more moves.
        :param dirichlet_x: percentage to favor dirichlet noise vs. prior estimation. Smaller means less noise
        """
        self.__id: uuid.UUID = uuid.uuid4()
        self._min_reward: float = min_reward
        self._pb_c_base: float = pb_c_base
        self._pb_c_init: float = pb_c_init
        self._dirichlet_noise: bool = dirichlet_noise
        self._dirichlet_alpha: float = dirichlet_alpha
        self._dirichlet_x: float = dirichlet_x
    
    @property
    def id(self) -> uuid.UUID:
        return self.__id
    
    @property
    def min_reward(self) -> float:
        return self._min_reward
    
    @property
    def pb_c_base(self) -> float:
        return self._pb_c_base
    
    @property
    def pb_c_init(self) -> float:
        return self._pb_c_init
    
    @property
    def dirichlet_noise(self) -> bool:
        return self._dirichlet_noise
    
    @property
    def dirichlet_alpha(self) -> float:
        return self._dirichlet_alpha
    
    @property
    def dirichlet_x(self) -> float:
        return self._dirichlet_x
    
    @abstractmethod
    def policy_predictions(self, policy_inputs_with_children):
        pass
    
    @abstractmethod
    def construct_feature_matrices(self, node: AlphaZeroNode):
        pass
    
    @abstractmethod
    def compute_reward(self, node: AlphaZeroNode) -> float:
        pass
