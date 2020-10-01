import psycopg2

# Initialize PostgreSQL tables

dbparams = {
    'dbname': 'bde',
    'port': 5432,
    'host': 'yuma.hpc.nrel.gov',
    'user': 'rlops',
    'password': 'jTeL85L!',
    'options': f'-c search_path=rl',
}

## This creates the table used to store the rewards
## But, we don't want this to run every time we run the script, 
## just keeping it here as a reference

with psycopg2.connect(**dbparams) as conn:
    with conn.cursor() as cur:
        cur.execute("""
        DROP TABLE IF EXISTS {table}_reward;
        
        CREATE TABLE {table}_reward (
            id serial PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            real_reward real,
            smiles varchar(50) UNIQUE,
            atom_type varchar(2),
            buried_vol real,
            max_spin real,
            atom_index int
            );
            
        DROP TABLE IF EXISTS {table}_replay;
        
        CREATE TABLE {table}_replay (
            id serial PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            experiment_id varchar(50),
            gameid varchar(8),        
            smiles varchar(50),
            final_smiles varchar(50),
            ranked_reward real,
            position int,
            data BYTEA); 

        DROP TABLE IF EXISTS {table}_game;
        
        CREATE TABLE {table}_game (
            id serial PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            experiment_id varchar(50),
            gameid varchar(8),   
            real_reward real);          
            """.format(table=config.sql_basename))