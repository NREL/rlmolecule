
import tensorflow as tf
import sys
import time
import subprocess


time.sleep(10)

# Copy the policy model to the rollout nodes
policy_checkpoint_dir = sys.argv[1]
rollout_nodes_file = sys.argv[2]
# number of seconds to wait to check for a new policy model
game_count_delay = 20
checkpoint = None

rollout_nodes = set()
with open(rollout_nodes_file, 'r') as f:
    for line in f:
        rollout_nodes.add(line.rstrip())

while True:
    new_checkpoint = tf.train.latest_checkpoint(policy_checkpoint_dir)
    if new_checkpoint != checkpoint:
        checkpoint = new_checkpoint
        print("Copying checkpoint to rollout nodes")
        checkpoint_file = f"{policy_checkpoint_dir}/checkpoint"

        for node in rollout_nodes:
            command = f"scp {checkpoint}* {checkpoint_file} {node}:/tmp/scratch/"
            print(f"running {command}")
            try:
                subprocess.check_call(command, shell=True)
            except subprocess.CalledProcessError as e:
                print(f"CalledProcessError {e}")

    else:
        print(f"Waiting for new checkpoint from {policy_checkpoint_dir}")
    time.sleep(game_count_delay)
