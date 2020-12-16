import sys
import uuid
sys.path.append('..')

import pandas as pd
import psycopg2
import rdkit
from rdkit import Chem
from rdkit import DataStructs
import networkx as nx

from rlmolecule.node import Node
from rlmolecule.mcts import run_mcts

radical_fps = pd.read_pickle('/projects/rlmolecule/pstjohn/q2_milestone/binary_fps.p.gz').apply(
    DataStructs.CreateFromBinaryText)
radicals = pd.read_csv('/projects/rlmolecule/pstjohn/q2_milestone/radicals.csv.gz')['0']

radical_set = set(radicals)

dbparams = {
    'dbname': 'bde',
    'port': 5432,
    'host': 'yuma.hpc.nrel.gov',
    'user': 'rlops',
    'password': '********',
    'options': f'-c search_path=rl',
}

## This creates the table used to store the rewards
## But, we don't want this to run every time we run the script, 
## just keeping it here as a reference

# with psycopg2.connect(**dbparams) as conn:
#     with conn.cursor() as cur:
#         cur.execute("""
#         DROP TABLE IF EXISTS Q2Reward;
        
#         CREATE TABLE Q2Reward (
#             id serial PRIMARY KEY,
#             time timestamp DEFAULT CURRENT_TIMESTAMP,
#             reward real,
#             smiles varchar(50) UNIQUE
#             );
            
#         DROP TABLE IF EXISTS Q2Replay;
        
#         CREATE TABLE Q2Replay (
#             id serial PRIMARY KEY,
#             time timestamp DEFAULT CURRENT_TIMESTAMP,
#             gameid varchar(8),        
#             smiles varchar(50),
#             reward real,
#             position int,
#             data BYTEA);          
#             """)

class SimilarityNode(Node):
    
    def get_reward(self):
            
        if self.smiles in radical_set:
            return -1.
        
        target_fp = Chem.RDKFingerprint(self)
        max_similarity = max(DataStructs.BulkTanimotoSimilarity(target_fp, radical_fps.values))
        
        # Here's how I'm doing the reward saving. Not the most efficient method (might want
        # to check if the reward exists before calculating) but this is a pretty fast 
        # reward calculation
        with psycopg2.connect(**dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO Q2Reward (smiles, reward) values (%s, %s)
                on conflict (smiles) do nothing;""", (self.smiles, max_similarity))
            
        if (max_similarity > 0.7) & (max_similarity < 1.0):
            return 1.
        else:
            return -1.
        
def run_game():
    """Run game, saving the results in a Postgres replay buffer"""

    gameid = uuid.uuid4().hex[:8]
    print(f'starting game {gameid}', flush=True)

    G = nx.DiGraph()
    start = SimilarityNode(rdkit.Chem.MolFromSmiles('C'), graph=G)
    G.add_node(start)

    game = list(run_mcts(G, start))
    reward = game[-1].reward

    with psycopg2.connect(**dbparams) as conn:
        for i, node in enumerate(game[:-1]):
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO Q2Replay (gameid, smiles, reward, position, data) values %s;""",
                    ((gameid, node.smiles, reward, i, node.get_action_inputs_as_binary()),))

    print(f'finishing game {gameid}', flush=True)
            

if __name__ == "__main__":    
        while True:
            run_game()
