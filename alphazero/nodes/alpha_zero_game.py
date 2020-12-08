import logging
import uuid
from abc import abstractmethod

import tensorflow as tf
import numpy as np

from alphazero.nodes.alphazero_node import AlphaZeroNode

logger = logging.getLogger(__name__)


class AlphaZeroGame:
    
    def __init__(self, config: any) -> None:
        self.__config: any = config
        self.__id: uuid.UUID = uuid.uuid4()
        
        self._policy_trainer = None
        self._policy_model = None
        self._policy_predictions = None
    
    @property
    def config(self) -> any:
        return self.__config
    
    @property
    def id(self) -> uuid.UUID:
        return self.__id
        
    def _setup(self, policy_trainer, policy_model, policy_predictions) -> None:
        self._policy_trainer = policy_trainer
        self._policy_model = policy_model
        self._policy_predictions = policy_predictions
        
        latest = tf.train.latest_checkpoint(self.config.checkpoint_filepath)
        if latest:
            self._policy_trainer.load_weights(latest)
            logger.info(f'{self.id}: loaded checkpoint {latest}')
        else:
            logger.info(f'{self.id}: no checkpoint found')
    
    # @abstractmethod
    # def
