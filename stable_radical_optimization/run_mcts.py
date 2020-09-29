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
import alphazero.mod as mod
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

reward_table = config.sql_basename + "_reward"
replay_table = config.sql_basename + "_replay"
game_table = config.sql_basename + "_game"

## This creates the table used to store the rewards
## But, we don't want this to run every time we run the script, 
## just keeping it here as a reference

with psycopg2.connect(**dbparams) as conn:
    with conn.cursor() as cur:
        cur.execute("""
        DROP TABLE IF EXISTS {table0};
        
        CREATE TABLE {table0} (
            id serial PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            reward real,
            smiles varchar(50) UNIQUE,
            atom_type varchar(2),
            buried_vol real,
            max_spin real,
            atom_index int
            );
            
        DROP TABLE IF EXISTS {table1};
        
        CREATE TABLE {table1} (
            id serial PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            experiment_id varchar(50),
            gameid varchar(8),        
            smiles varchar(50),
            final_smiles varchar(50),
            ranked_reward real,
            position int,
            data BYTEA); 

        DROP TABLE IF EXISTS {table2};
        
        CREATE TABLE {table2} (
            id serial PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            experiment_id varchar(50),
            reward real);          
            """.format(table0=reward_table, table1=replay_table, table2=game_table))

class StabilityNode(Node):
    
    def get_reward(self):
        
        with psycopg2.connect(**dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute("select reward from {table} where smiles = %s".format(table=reward_table), (self.smiles,))
                result = cur.fetchone()
        
        if result:
            # Probably would put RR code here?
            return result[0]
        
        # Node is outside the domain of validity
        elif ((self.policy_inputs['atom'] == 1).any() | 
              (self.policy_inputs['bond'] == 1).any()):
            return 0.
        
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
                        INSERT INTO {table} 
                        (experiment_id, reward) 
                        values (%s, %s);""".format(table=game_table), (
                            config.experiment_id, float(reward)))
                
                df = pd.read_sql_query("""
                select reward from {table}
                """.format(table=game_table), conn)

                if len(df.index) < config.batch_size:
                    reward = np.random.choice([-1.,1.])
                else:
                    param = {config.ranked_reward_alpha, config.batch_size}
                    r_alpha = pd.read_sql_query("""
                        select percentile_cont(%s) within group (order by reward) from (
                            select reward 
                            from {table} 
                            order by id desc limit %s) as finals  
                        """.format(table=game_table), conn, params=param)
                    if reward > r_alpha['percentile_cont'][0]:
                        reward = 1.
                    elif reward < r_alpha['percentile_cont'][0]:
                        reward = -1.
                    else:
                        reward = np.random.choice([-1.,1.])
            
            with psycopg2.connect(**dbparams) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO {table} 
                        (smiles, reward, atom_type, buried_vol, max_spin, atom_index) 
                        values (%s, %s, %s, %s, %s, %s);""".format(table=reward_table), (
                            self.smiles, float(reward), atom_type,
                            float(spin_buried_vol), float(max_spin), atom_index))
            
            return reward

        
def run_game():
    """Run game, saving the results in a Postgres replay buffer"""

    gameid = uuid.uuid4().hex[:8]
    print(f'starting game {gameid}', flush=True)

    G = Game(StabilityNode, 'C', checkpoint_dir='.')

    game = list(G.run_mcts())
    reward = game[-1].reward

    with psycopg2.connect(**dbparams) as conn:
        for i, node in enumerate(game[:-1]):
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO {table} 
                    (experiment_id, gameid, smiles, final_smiles, ranked_reward, position, data) values %s;""".format(table=replay_table),
                    ((config.experiment_id, gameid, node.smiles, game[-1].smiles, reward, i,
                      node.get_action_inputs_as_binary()),))

    print(f'finishing game {gameid}', flush=True)
            

if __name__ == "__main__":    
        while True:
            run_game()
