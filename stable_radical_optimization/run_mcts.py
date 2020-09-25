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

from alphazero.node import Node
from alphazero.game import Game

import tensorflow as tf
import nfp

model = tf.keras.models.load_model(
    '/projects/rlmolecule/pstjohn/models/20200923_radical_stability_model')

dbparams = {
    'dbname': 'bde',
    'port': 5432,
    'host': 'yuma.hpc.nrel.gov',
    'user': 'rlops',
    'password': '***REMOVED***',
    'options': f'-c search_path=rl',
}

## This creates the table used to store the rewards
## But, we don't want this to run every time we run the script, 
## just keeping it here as a reference

# with psycopg2.connect(**dbparams) as conn:
#     with conn.cursor() as cur:
#         cur.execute("""
#         DROP TABLE IF EXISTS StableRewardPSJ;
        
#         CREATE TABLE StableRewardPSJ (
#             id serial PRIMARY KEY,
#             time timestamp DEFAULT CURRENT_TIMESTAMP,
#             reward real,
#             smiles varchar(50) UNIQUE,
#             atom_type varchar(2),
#             buried_vol real,
#             max_spin real,
#             atom_index int
#             );
            
#         DROP TABLE IF EXISTS StableReplayPSJ;
        
#         CREATE TABLE StableReplayPSJ (
#             id serial PRIMARY KEY,
#             time timestamp DEFAULT CURRENT_TIMESTAMP,
#             gameid varchar(8),        
#             smiles varchar(50),
#             reward real,
#             position int,
#             data BYTEA);          
#             """)

class StabilityNode(Node):
    
    def get_reward(self):
        
        with psycopg2.connect(**dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute("select reward from StableRewardPSJ where smiles = %s", (self.smiles,))
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
                    INSERT INTO StableRewardPSJ 
                    (smiles, reward, atom_type, buried_vol, max_spin, atom_index) 
                    values (%s, %s, %s, %s, %s, %s);""", (
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
                    """INSERT INTO StableReplayPSJ 
                    (gameid, smiles, final_smiles, reward, position, data) values %s;""",
                    ((gameid, node.smiles, game[-1].smiles, reward, i,
                      node.get_action_inputs_as_binary()),))

    print(f'finishing game {gameid}', flush=True)
            

if __name__ == "__main__":    
        while True:
            run_game()
