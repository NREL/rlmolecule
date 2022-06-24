
import os
import sys
import time
import subprocess
import pandas as pd
import tensorflow as tf

from rlmolecule.sql import Base, Session
from rlmolecule.sql.tables import GameStore, RewardStore
from rlmolecule.sql.run_config import RunConfig

time.sleep(10)


def get_states_seen(session, run_id, reward_cutoff=0.25):
    start = time.time()
    df = pd.read_sql(session.query(RewardStore)
                            .filter_by(run_id=run_id)
                            .statement, session.bind)
    print(f"{time.time() - start:0.2f} seconds to load rewards")

    df['state'] = df['data'].apply(lambda x: x['state_repr'])
    df['terminal'] = df['data'].apply(lambda x: str(x['terminal']).lower() == "true")
    states = set(df[(df['terminal']) & (df['reward'] >= reward_cutoff)]['state'].values)

    return states


run_config = RunConfig(sys.argv[1])
run_id = run_config.run_id
train_config = run_config.train_config

engine = run_config.start_engine()
Base.metadata.create_all(engine, checkfirst=True)
Session.configure(bind=engine)
session = Session()

policy_checkpoint_dir = train_config.get('policy_checkpoint_dir',
                                         'policy_checkpoints')

# Copy the policy model to the rollout nodes
rollout_nodes_file = os.path.dirname(policy_checkpoint_dir) + '/.rollout_nodes.txt'
# also copy the states seen to the rollout nodes
states_seen_file = os.path.dirname(policy_checkpoint_dir) + '/states_seen.csv.gz'

rollout_nodes = set()
with open(rollout_nodes_file, 'r') as f:
    for line in f:
        rollout_nodes.add(line.rstrip())

# number of seconds to wait to check for a new policy model
game_count_delay = 20
checkpoint = None
while True:
    new_checkpoint = tf.train.latest_checkpoint(policy_checkpoint_dir)
    if new_checkpoint != checkpoint:
        checkpoint = new_checkpoint
        print("Copying checkpoint to rollout nodes")
        checkpoint_file = f"{policy_checkpoint_dir}/checkpoint"

        for node in rollout_nodes:
            command = f"rsync {checkpoint}* {checkpoint_file} {node}:/tmp/scratch/"
            print(f"running {command}")
            try:
                subprocess.check_call(command, shell=True)
            except subprocess.CalledProcessError as e:
                print(f"CalledProcessError {e}")

    else:
        print(f"Waiting for new checkpoint from {policy_checkpoint_dir}")

#    # now load the rewards, and write the states seen to a file
#    #states_seen = get_states_seen(session, run_id)
#    # For now, just get all states
#    states_seen = get_states_seen(session, run_id, reward_cutoff=0)
#    df = pd.DataFrame(states_seen, columns=['states'])
#    # also write a csv for now to compare size, load time
#    df.to_csv(states_seen_file, index=False)
#    for node in rollout_nodes:
#        command = f"rsync {states_seen_file} {node}:/tmp/scratch/"
#        print(f"running {command}")
#        try:
#            subprocess.check_call(command, shell=True)
#        except subprocess.CalledProcessError as e:
#            print(f"CalledProcessError {e}")

    time.sleep(game_count_delay)
