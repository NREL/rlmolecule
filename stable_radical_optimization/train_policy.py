import os
import sys
import time
sys.path.append('..')

import psycopg2
import pandas as pd
import tensorflow as tf

import alphazero.config as config
from alphazero.policy import build_policy_trainer
from alphazero.policy_data import create_dataset

import stable_rad_config


def psql_generator():
    """ A python generator to yield rows from the Postgres database. Note, here I'm deferring
    the actual parsing of the binary data to a later function, which we can hopefully parallelize.
    
    The SQL command here selects 100 random game states, selected from the (unique) 100 most recent
    games (id is the row-id, always increasing with newer games; gameid is a unique game identifier)
    
    Essentially when this runs out; it should get re-called to grab new data. 
    """
        
    with psycopg2.connect(**config.dbparams) as conn:
            
        df = pd.read_sql_query("""
        with recent_replays as (
            select * from rl.stablepsj_replay where gameid in (
                select gameid from {}_game where experiment_id = %s order by id desc limit %s))

        select distinct on (gameid) id, ranked_reward, data
            from recent_replays order by gameid, random();
        """.format(config.sql_basename), conn, params=(config.experiment_id, config.buffer_max_size,))
        
        for _, row in df.iterrows():
            yield (row.data.tobytes(), row.ranked_reward)
            
            
dataset = create_dataset(psql_generator)
policy_trainer = build_policy_trainer()

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(config.checkpoint_filepath, 'policy.{epoch:02d}'),
    save_best_only=False, save_weights_only=True)

# wait to start training until enough games have occurred
while len(list(psql_generator())) < config.buffer_min_size:
    time.sleep(60)

policy_trainer.fit(dataset, steps_per_epoch=config.steps_per_epoch,
                   epochs=int(1E4), callbacks=[model_checkpoint])