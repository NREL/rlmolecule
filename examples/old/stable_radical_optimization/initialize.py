import argparse
import sys
sys.path.append("../..")

import psycopg2

import molecule_game.config as config

# Initialize PostgreSQL tables

parser = argparse.ArgumentParser(description='Initialize the postgres tables.')
parser.add_argument("--drop", action='store_true', help="whether to drop existing tables, if found")
args = parser.parse_args()

## This creates the table used to store the rewards
## But, we don't want this to run every time we run the script, 
## just keeping it here as a reference

with psycopg2.connect(**config.dbparams) as conn:    
    with conn.cursor() as cur:
        
        if args.drop:
            cur.execute("""
            DROP TABLE IF EXISTS {table}_reward;
            DROP TABLE IF EXISTS {table}_replay;
            DROP TABLE IF EXISTS {table}_game;
            """.format(table=config.sql_basename))
            
        cur.execute("""
        CREATE TABLE IF NOT EXISTS {table}_reward (
            smiles varchar(50) PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            real_reward real,
            atom_type varchar(2),
            buried_vol real,
            max_spin real,
            atom_index int
            );
            
        CREATE TABLE IF NOT EXISTS {table}_replay (
            id serial PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            experiment_id varchar(50),
            gameid varchar(8),        
            smiles varchar(50),
            final_smiles varchar(50),
            ranked_reward real,
            position int,
            data BYTEA); 

        CREATE TABLE IF NOT EXISTS {table}_game (
            id serial PRIMARY KEY,
            time timestamp DEFAULT CURRENT_TIMESTAMP,
            experiment_id varchar(50),
            gameid varchar(8),
            real_reward real,
            final_smiles varchar(50));          
            """.format(table=config.sql_basename))
