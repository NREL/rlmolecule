import os
import sys
import logging

sys.path.append('../..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import psycopg2

import molecule_game.config as config
from rlmolecule.tree_search.graph_search import GraphSearch

import tensorflow as tf

model = tf.keras.models.load_model(
    config.reward_model_path,
    compile=False)

@tf.function(experimental_relax_shapes=True)                
def predict(inputs):
    return model.predict_step(inputs)

def get_ranked_rewards(reward):

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
    

class StabilityNode(Node):
      
    def get_reward(self):
        
        with psycopg2.connect(**config.dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select real_reward from {table}_reward where smiles = %s".format(
                        table=config.sql_basename), (self.smiles,))
                result = cur.fetchone()
        
        if result:
            # Probably would put RR code here?
            rr = get_ranked_rewards(result[0])
            self._true_reward = result[0]
            return rr
        
        # Node is outside the domain of validity
        elif ((self.policy_inputs['atom'] == 1).any() | 
              (self.policy_inputs['bond'] == 1).any()):
            self._true_reward = 0.
            return config.min_reward
        
        else:           
            
            spins, buried_vol = predict(
                {key: tf.constant(np.expand_dims(val, 0))
                 for key, val in self.policy_inputs.items()})
        
            spins = spins.numpy().flatten()
            buried_vol = buried_vol.numpy().flatten()

            atom_index = int(spins.argmax())
            max_spin = spins[atom_index]
            spin_buried_vol = buried_vol[atom_index]
            
            atom_type = self.GetAtomWithIdx(atom_index).GetSymbol()

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
                            self.smiles, float(reward), atom_type, # This should be the real reward
                            float(spin_buried_vol), float(max_spin), atom_index))
            
            rr = get_ranked_rewards(reward)
            self._true_reward = reward
            return rr

        
def run_game():
    """Run game, saving the results in a Postgres replay buffer"""

    G = GraphSearch(StabilityNode, 'C')
    logging.info(f"Starting {G.id}")
    
    game = list(G.run_mcts())
    reward = game[-1].reward # here it returns the ranked reward
    
    try:
        terminal_true_reward = float(game[-1]._true_reward)
        
    except AttributeError:
        # This is hacky until we have a better separation of `true_reward`
        # and `ranked_reward`. This can happen if a node doesn't have any
        # children, but still gets chosen as the final state.
        terminal_true_reward = 0.
    
    logging.info(f"Finishing {G.id}: true_reward={terminal_true_reward:.1f}, ranked reward={reward}")
    
    with psycopg2.connect(**config.dbparams) as conn:

        with conn.cursor() as cur:
            cur.execute(
                """                    
                INSERT INTO {table}_game
                (experiment_id, gameid, real_reward, final_smiles) values (%s, %s, %s, %s);
                """.format(table=config.sql_basename), (
                    config.experiment_id, G.id, terminal_true_reward, game[-1].smiles))

        for i, node in enumerate(game[:-1]):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO {table}_replay
                    (experiment_id, gameid, smiles, final_smiles, ranked_reward, position, data) 
                    values (%s, %s, %s, %s, %s, %s, %s);
                    """.format(table=config.sql_basename), (
                        config.experiment_id, G.id, node.smiles, game[-1].smiles, reward, i,
                        node._policy_data))
                            

if __name__ == "__main__":    
      
    while True:
        run_game()
