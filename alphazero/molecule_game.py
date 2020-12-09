import logging
import os
from pprint import pprint

import tensorflow as tf
from rdkit.Chem.rdmolfiles import MolFromSmiles

from alphazero.nodes.alpha_zero_game import AlphaZeroGame
from alphazero.nodes.alphazero_node import AlphaZeroNode
from alphazero.nodes.networkx_node_memoizer import NetworkXNodeMemoizer
from alphazero.policy import build_policy_trainer
from alphazero.preprocessor import (
    MolPreprocessor,
    atom_featurizer,
    bond_featurizer,
    )
from molecule_graph.molecule_node import MoleculeNode

logger = logging.getLogger(__name__)

default_preprocessor = MolPreprocessor(atom_features=atom_featurizer,
                                       bond_features=bond_featurizer,
                                       explicit_hs=False)

default_preprocessor.from_json(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'preprocessor.json'))


class MoleculeGame(AlphaZeroGame):
    
    def __init__(self, config: any, start_smiles: str, preprocessor: MolPreprocessor = default_preprocessor) -> None:
        super().__init__(
            config.min_reward,
            config.pb_c_base,
            config.pb_c_init,
            config.dirichlet_noise,
            config.dirichlet_alpha,
            config.dirichlet_x,
            )
        self._config = config
        self._preprocessor: MolPreprocessor = preprocessor
        self._graph_memoizer: NetworkXNodeMemoizer = NetworkXNodeMemoizer()
        
        # noinspection PyTypeChecker
        self._start: AlphaZeroNode = \
            self._graph_memoizer.memoize(
                AlphaZeroNode(
                    MoleculeNode(self, MolFromSmiles(start_smiles)),
                    self))
        
        pprint(self._start.graph_node)
        
        self._policy_trainer = build_policy_trainer()
        self._policy_model = self._policy_trainer.layers[-1].policy_model
        self._policy_predictions = tf.function(experimental_relax_shapes=True)(self._policy_model.predict_step)
        self.load_from_checkpoint()  # TODO: does this ever do anything?
    
    def construct_feature_matrices(self, node: AlphaZeroNode):
        return self._preprocessor.construct_feature_matrices(node.graph_node)
    
    def policy_predictions(self, policy_inputs_with_children):
        return self._policy_predictions(policy_inputs_with_children)
    
    def load_from_checkpoint(self):
        latest = tf.train.latest_checkpoint(self._config.checkpoint_filepath)
        if latest:
            self._policy_trainer.load_weights(latest)
            logger.info(f'{self.id}: loaded checkpoint {latest}')
        else:
            logger.info(f'{self.id}: no checkpoint found')
    