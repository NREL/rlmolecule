import logging
import os
from pprint import pprint
from typing import (
    Iterator,
    Optional,
    )

import psycopg2
import tensorflow as tf
from rdkit.Chem.rdmolfiles import MolFromSmiles

from alphazero.alpha_zero_game import AlphaZeroGame
from alphazero.alphazero_node import AlphaZeroNode
from alphazero.networkx_node_memoizer import NetworkXNodeMemoizer
from molecule_game.molecule_policy import build_policy_trainer
from molecule_game.mol_preprocessor import (
    MolPreprocessor,
    atom_featurizer,
    bond_featurizer,
    )
from molecule_game.stable_radical_optimization.stable_radical_optimization_node import StableRadicalOptimizationNode
from run_mcts import predict
import numpy as np
logger = logging.getLogger(__name__)

default_preprocessor = MolPreprocessor(atom_features=atom_featurizer,
                                       bond_features=bond_featurizer,
                                       explicit_hs=False)

default_preprocessor.from_json(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../molecule_game/preprocessor.json'))


class StableRadicalOptimizationGame(AlphaZeroGame):
    """
    An AlphaZeroGame that implements a stable radical optimization search.
    """
    
    def __init__(
            self,
            config: any,
            start_smiles: str,
            preprocessor: MolPreprocessor = default_preprocessor,
            ) -> None:
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
                    StableRadicalOptimizationNode(self, MolFromSmiles(start_smiles), False),
                    self))
        
        pprint(self._start.graph_node)
        
        self._policy_trainer = build_policy_trainer()
        self._policy_model = self._policy_trainer.layers[-1].policy_model
        self._policy_predictions = tf.function(experimental_relax_shapes=True)(self._policy_model.predict_step)
        self.load_from_checkpoint()  # TODO: does this ever do anything?
    
    @property
    def config(self) -> any:
        return self._config
    
    def construct_feature_matrices(self, node: AlphaZeroNode):
        return self._preprocessor.construct_feature_matrices(node.graph_node.molecule)
    
    def policy_predictions(self, policy_inputs_with_children):
        return self._policy_predictions(policy_inputs_with_children)
    
    def load_from_checkpoint(self):
        latest = tf.train.latest_checkpoint(self._config.checkpoint_filepath)
        if latest:
            self._policy_trainer.load_weights(latest)
            logger.info(f'{self.id}: loaded checkpoint {latest}')
        else:
            logger.info(f'{self.id}: no checkpoint found')
    
    def run_mcts(self, num_simulations: Optional[int] = None, explore: bool = True) -> Iterator['AlphaZeroNode']:
        num_simulations = self._config.num_simulations if num_simulations is None else num_simulations
        return self._start.run_mcts(num_simulations, explore)
    
    def compute_reward(self, node: AlphaZeroNode) -> float:
        config = self.config
        with psycopg2.connect(**config.dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select real_reward from {table}_reward where smiles = %s".format(
                        table=config.sql_basename), (node.graph_node.smiles,))
                result = cur.fetchone()
        
        if result:
            # Probably would put RR code here?
            rr = self.get_ranked_rewards(result[0])
            self._true_reward = result[0]
            return rr
        
        # Node is outside the domain of validity
        elif ((node.policy_inputs['atom'] == 1).any() |
              (node.policy_inputs['bond'] == 1).any()):
            self._true_reward = 0.
            return config.min_reward
        
        else:
            
            spins, buried_vol = predict(
                {key: tf.constant(np.expand_dims(val, 0))
                 for key, val in node.policy_inputs.items()})
            
            spins = spins.numpy().flatten()
            buried_vol = buried_vol.numpy().flatten()
            
            atom_index = int(spins.argmax())
            max_spin = spins[atom_index]
            spin_buried_vol = buried_vol[atom_index]
            
            atom_type = node.graph_node.molecule.GetAtomWithIdx(atom_index).GetSymbol()
            
            # This is a bit of a placeholder; but the range for spin is about 1/50th that
            # of buried volume.
            reward = (1 - max_spin) * 50 + spin_buried_vol
            
            with psycopg2.connect(**config.dbparams) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO {table}_reward
                        (smiles, real_reward, atom_type, buried_vol, max_spin, atom_index)
                        values (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;""".format(table=config.sql_basename), (
                        node.graph_node.smiles, float(reward), atom_type,  # This should be the real reward
                        float(spin_buried_vol), float(max_spin), atom_index))
            
            rr = self.get_ranked_rewards(reward)
            
            # TODO: do we want to store this here?
            # node._true_reward = reward
            return rr
    
    def get_ranked_rewards(self, reward: float) -> float:
        config = self.config
        with psycopg2.connect(**config.dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute("select count(*) from {table}_game where experiment_id = %s;".format(
                    table=config.sql_basename), (config.experiment_id,))
                n_games = cur.fetchone()[0]
            
            if n_games < config.reward_buffer_min_size:
                # Here, we don't have enough of a game buffer
                # to decide if the move is good or not
                logging.debug(f"ranked_reward: not enough games ({n_games})")
                return np.random.choice([0., 1.])
            
            else:
                with conn.cursor() as cur:
                    cur.execute("""
                            select percentile_disc(%s) within group (order by real_reward)
                            from (select real_reward from {table}_game where experiment_id = %s
                                  order by id desc limit %s) as finals
                            """.format(table=config.sql_basename),
                                (config.ranked_reward_alpha, config.experiment_id, config.reward_buffer_max_size))
                    
                    r_alpha = cur.fetchone()[0]
                
                logging.debug(f"ranked_reward: r_alpha={r_alpha}, reward={reward}")
                
                if np.isclose(reward, r_alpha):
                    return np.random.choice([0., 1.])
                
                elif reward > r_alpha:
                    return 1.
                
                elif reward < r_alpha:
                    return 0.
