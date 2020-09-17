from rollout import play_game, Network

network = Network('.')
game = play_game(network)
data = game.get_data()

import numpy as np
import io
from tqdm import tqdm

# Save the data dict to a binary sequence
with io.BytesIO() as f:
    np.savez_compressed(f, **data)
    binary_data = f.getvalue()
    
del data

# Lead the data dict from a binary sequence
with io.BytesIO(binary_data) as f:
    data = dict(np.load(f, allow_pickle=True).items())

print(data['mol_smiles'])

import psycopg2

dbparams = {
    'dbname': 'bde',
    'port': 5432,
    'host': 'yuma.hpc.nrel.gov',
    'user': 'rlops',
    'password': 'jTeL85L!',
    'options': f'-c search_path=rl',
}

    
with psycopg2.connect(**dbparams) as conn: # func connect creates new db session, returns a new connection instance
    with conn.cursor() as cur: # in the new connection instance, create cursor to execute db commands/queries. sends commands to db using execute
        cur.execute("""
        DROP TABLE IF EXISTS TestReplay;
        
        CREATE TABLE TestReplay (
            gameid serial PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            data BYTEA);
            
        INSERT INTO TestReplay (data) VALUES (%s)
        """, (binary_data,)) # use placeholders %s to pass parameters to SQL statement

import pandas as pd

with psycopg2.connect(**dbparams) as conn:
    df = pd.read_sql_query("""
    SELECT * from TestReplay;
    """, conn)
    
print(df)

def read_game_data(binary_data):
    """Parse our binary game format. We can use pandas.apply to map 
    this over an entire column if we want """
    with io.BytesIO(df.iloc[0].data) as f:
        data = dict(np.load(f, allow_pickle=True).items())
        
    return data

print(df.data.apply(read_game_data))

def save_game_postgresql(game, conn):
    """ Just combines a bunch of the last few functions together """
    
    data = game.get_data()
    
    with io.BytesIO() as f:
        np.savez_compressed(f, **data)
        binary_data = f.getvalue()
        
    with conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO TestReplay (data) VALUES (%s)",
                        (binary_data,))

# I think it makes sense to keep this connection alive, 
# as long as the games are somewhat fast.. But we could certainly
# close it and re-open even 

with psycopg2.connect(**dbparams) as conn:
    for _ in tqdm(range(10)):
        game = play_game(network)
        save_game_postgresql(game, conn)

# Here's how we'd grab the most recent batch of games from the buffer

with psycopg2.connect(**dbparams) as conn:
    df = pd.read_sql_query("""
    SELECT * from TestReplay order by time desc limit 4;
    """, conn)

print(df)