import argparse
import os

import pandas as pd
from tqdm import tqdm

import ray
from ray.tune.registry import register_env
from ray.tune import Analysis
import ray.rllib.agents.ppo as ppo

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def main(args):

    ray.init(_node_ip_address=args["node_ip_address"], num_cpus=1, num_gpus=0)

    # values that get used more than once
    restore_dir = args["restore_dir"]
    chkpt = args["checkpoint"]

    a = Analysis(restore_dir)
    config = a.get_best_config("episode_reward_mean", mode="max")
    config["num_workers"] = 1
    config["num_gpus"] = 0

    print("CONFIG", config)

    env_name = "molecule_graph_problem"
    
    from optimize_qed import env_creator

    _ = register_env(env_name, env_creator)
    config["env"] = env_name
    config.update(args)  # TODO: use all env_config args at train time so they are saved

    trainer = ppo.PPOTrainer(env=env_name, config=config)
    checkpoint = os.path.join(
        restore_dir, 
        "checkpoint_{}/checkpoint-{}".format(
            chkpt, int(chkpt)))
    trainer.restore(checkpoint)

    env = env_creator(config)

    all_rewards = []
    for _ in tqdm(range(args["num_episodes"])):
        obs = env.reset()
        actions = []
        rewards = []
        states = []
        episode_reward = 0. 
        done = False
        while not done:
            action = trainer.compute_actions(obs, explore=False)
            obs, rew, done, _ = env.step(action)
            rewards.append(rew)
            actions.append(action.copy())
            states.append(obs.copy())
            episode_reward += sum(list(rew.values()))
        all_rewards.append(episode_reward)
        print(env.state, episode_reward)

    print("rewards summary")
    df = pd.DataFrame(all_rewards)
    print(df.describe())


if __name__ == "__main__":

    from optimize_qed_argparser import parser

    parser.add_argument("--restore-dir", default=None, type=str)
    parser.add_argument("--checkpoint", default=None, type=str)

    args = vars(parser.parse_args())

    main(args)
