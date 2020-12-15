import os
import sys
import logging

sys.path.append('..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import psycopg2

import molecule_game.config as config
import stable_rad_config

from molecule_game.stable_radical_optimization.stable_radical_optimization_game import StableRadicalOptimizationGame
        
def run_game():
    """Run game, saving the results in a Postgres replay buffer"""

    game = StableRadicalOptimizationGame(stable_rad_config.config, 'C')
    logging.info(f"Starting {game.id}")
    
    game_history = list(game.run_mcts())
    reward = game_history[-1].reward # here it returns the ranked reward
    
    try:
        terminal_true_reward = float(game_history[-1]._true_reward)
        
    except AttributeError:
        # This is hacky until we have a better separation of `true_reward`
        # and `ranked_reward`. This can happen if a node doesn't have any
        # children, but still gets chosen as the final state.
        terminal_true_reward = 0.
    
    logging.info(f"Finishing {game.id}: true_reward={terminal_true_reward:.1f}, ranked reward={reward}")
    
    with psycopg2.connect(**config.dbparams) as conn:

        with conn.cursor() as cur:
            cur.execute(
                """                    
                INSERT INTO {table}_game
                (experiment_id, gameid, real_reward, final_smiles) values (%s, %s, %s, %s);
                """.format(table=config.sql_basename), (
                    config.experiment_id, game.id, terminal_true_reward, game_history[-1].smiles))

        for i, node in enumerate(game_history[:-1]):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO {table}_replay
                    (experiment_id, gameid, smiles, final_smiles, ranked_reward, position, data) 
                    values (%s, %s, %s, %s, %s, %s, %s);
                    """.format(table=config.sql_basename), (
                        config.experiment_id, game.id, node.smiles, game_history[-1].smiles, reward, i,
                        node._policy_data))
                            

if __name__ == "__main__":    
      
    while True:
        run_game()
