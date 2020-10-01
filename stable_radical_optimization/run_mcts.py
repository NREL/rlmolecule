import sys
import uuid
sys.path.append('..')

import numpy as np
import pandas as pd
import psycopg2
import rdkit
from rdkit import Chem
from rdkit import DataStructs
import networkx as nx

import alphazero.config as config
import stable_rad_config
from alphazero.node import Node
from alphazero.game import Game

import tensorflow as tf
import nfp

tf.logging.set_verbosity(tf.logging.ERROR)

model = tf.keras.models.load_model(
    '/projects/rlmolecule/pstjohn/models/20200923_radical_stability_model')

dbparams = {
    'dbname': 'bde',
    'port': 5432,
    'host': 'yuma.hpc.nrel.gov',
    'user': 'rlops',
    'password': 'jTeL85L!',
    'options': f'-c search_path=rl',
}

def get_ranked_rewards(reward, conn=None):

    if conn in None:
        conn = psycopg2.connect(**dbparams)
    
    with conn:
        n_rewards = pd.read_sql_query("""
        select count(*) from {table}_reward
        """.format(table=config.sql_basename), conn)

        if n_rewards < config.batch_size:
            return np.random.choice([-1.,1.])
        else:
            param = {config.ranked_reward_alpha, config.batch_size}
            r_alpha = pd.read_sql_query("""
                select percentile_cont(%s) within group (order by real_reward) from (
                    select real_reward 
                    from {table}_reward
                    order by id desc limit %s) as finals  
                """.format(table=config.sql_basename), conn, params=param)
            if reward > r_alpha['percentile_cont'][0]:
                return 1.
            elif reward < r_alpha['percentile_cont'][0]:
                return -1.
            else:
                return np.random.choice([-1.,1.])
    

class StabilityNode(Node):
    
    def get_reward(self):

        node = self.G.nodes[self]
        
        with psycopg2.connect(**dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute("select real_reward from {table}_reward where smiles = %s".format(table=config.sql_basename), (self.smiles,))
                result = cur.fetchone()
        
        if result:
            # Probably would put RR code here?
            rr = get_ranked_rewards(result[0])
            node._true_reward = result[0]
            return rr
        
        # Node is outside the domain of validity
        elif ((self.policy_inputs['atom'] == 1).any() | 
              (self.policy_inputs['bond'] == 1).any()):
            rr = get_ranked_rewards(0.)
            node._true_reward = 0.
            return rr
        
        else:           
            spins, buried_vol = model(
                {key: tf.constant(np.expand_dims(val, 0))
                 for key, val in self.policy_inputs.items()})
        
            spins = spins.numpy().flatten()
            buried_vol = buried_vol.numpy().flatten()

            atom_index = int(spins.argmax())
            max_spin = spins[atom_index]
            spin_buried_vol = buried_vol[atom_index]
            
            # Hacky solution until NN trained without H's finishes
            if atom_index >= self.GetNumAtoms():
                return 0.
            
            atom_type = self.GetAtomWithIdx(atom_index).GetSymbol()

            # This is a bit of a placeholder; but the range for spin is about 1/50th that
            # of buried volume.
            reward = (1 - max_spin) * 50 + spin_buried_vol
            
            with psycopg2.connect(**dbparams) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO {table}_reward
                        (smiles, real_reward, atom_type, buried_vol, max_spin, atom_index) 
                        values (%s, %s, %s, %s, %s, %s);""".format(table=config.sql_basename), (
                            self.smiles, float(reward), atom_type, # This should be the real reward
                            float(spin_buried_vol), float(max_spin), atom_index))
            
            rr = get_ranked_rewards(reward)
            node._true_reward = reward
            return rr

        
def run_game():
    """Run game, saving the results in a Postgres replay buffer"""

    gameid = uuid.uuid4().hex[:8]
    print(f'starting game {gameid}', flush=True)

    G = Game(StabilityNode, 'C', checkpoint_dir='.')

    game = list(G.run_mcts())
    reward = game[-1].reward # here it returns the ranked reward

    with psycopg2.connect(**dbparams) as conn:
        for i, node in enumerate(game[:-1]):
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO {table}_replay
                    (experiment_id, gameid, smiles, final_smiles, ranked_reward, position, data) values %s;
                    
                    INSERT INTO {table}_game
                    (experiment_id, gameid, real_reward) values (%s, %s);
                    """.format(table=config.sql_basename),
                    ((config.experiment_id, gameid, node.smiles, game[-1].smiles, reward, i,
                      node.get_action_inputs_as_binary()),config.experiment_id, gameid, float(game[-1]._true_reward))
                
    print(f'finishing game {gameid}', flush=True)
            

if __name__ == "__main__":    
        while True:
            run_game()
