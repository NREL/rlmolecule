import argparse
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import ray
from ray.tune.registry import register_env
from ray.tune import Analysis
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PCA_EMBED_SIZE = 1000


def env_creator(_):
    # print('begin make_env()')
    # print('******************* import tensorflow as tf - 1')
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    from examples.gym.mol_fp_gym.molecule_graph_problem import MoleculeGraphProblem
    from rlmolecule.graph_gym.graph_gym_env import GraphGymEnv
    from rlmolecule.molecule.builder.builder import MoleculeBuilder

    # Load the PCA model for embedding
    import pickle
    with open(os.path.join("pca-{}.p".format(PCA_EMBED_SIZE)), "rb") as f:
        pca = pickle.load(f)

    result = GraphGymEnv(
        MoleculeGraphProblem(
            MoleculeBuilder(max_atoms=6, min_atoms=1),
            pca
        )
    )
    return result

from rlmolecule.graph_gym.graph_gym_model import GraphGymModel

class ThisModel(GraphGymModel):
    def __init__(self,
                    obs_space,
                    action_space,
                    num_outputs,
                    model_config,
                    name,
                    **kwargs):
        # print('******************* import tensorflow as tf - 2')
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # from examples.gym.mol_fp_gym.molecule_model import MoleculeModel
        from examples.gym.mol_fp_gym.policy import policy_model

        super(ThisModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name,
            policy_model,
            **kwargs)


def main(args):

    ray.init(_node_ip_address=args["ip"], num_cpus=1)

    # values that get used more than once
    restore_dir = args["restore_dir"]
    chkpt = args["checkpoint"]

    a = Analysis(restore_dir)
    config = a.get_best_config("episode_reward_mean", mode="max")
    config["num_workers"] = 1
    config["num_gpus"] = 0

    env_name = 'mol_fp_pca={}'.format(PCA_EMBED_SIZE)
    _ = register_env(env_name, env_creator)

    ModelCatalog.register_custom_model('molecule_graph_problem_model', ThisModel)

    for reload_try in range(10):
        trainer = ppo.PPOTrainer(env=env_name, config=config)
        checkpoint = os.path.join(
            restore_dir, 
            "checkpoint_{}/checkpoint-{}".format(
                chkpt, int(chkpt)))
        trainer.restore(checkpoint)
        policy = trainer.get_policy()
        print("reload try", reload_try, policy.get_weights())

    tic = time.time()
    all_rewards = []
    env = env_creator(None)

    return

    for ep in range(args["num_episodes"]):
        obs = env.reset()
        print("first obs=", obs["action_observations"][0]["fingerprint"][:10])
        actions = []
        rewards = []
        states = []
        episode_reward = 0.
        step = 0
        done = False
        print("EPISODE {}/{}".format(ep+1, args["num_episodes"], step))
        while not done:
            _actions = trainer.compute_single_action(obs, full_fetch=True, explore=False)
            action = trainer.compute_single_action(obs, explore=False)
            probs = _actions[2]["action_dist_inputs"]
            print("step={}, action={}, action logits={}".format(step+1, _actions[0], probs[probs > -999]))
            obs, rew, done, meta = env.step(action)
            #print("first obs=", obs["action_observations"][0])
            print("applied action={}".format(action))
            print("meta={}".format(meta))
            rewards.append(rew)
            actions.append(action)
            states.append(obs)
            episode_reward += rew
            step += 1
        all_rewards.append(episode_reward)
        print("total reward={}".format(all_rewards[-1]))
        print("elapsed, {}\n".format(time.time() - tic))
       
    print("rewards summary")
    df = pd.DataFrame(all_rewards)
    print(df.describe())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-dir", default=None, type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--num-episodes", default=1, type=int)
    parser.add_argument("--ip", default="127.0.0.1", type=str)
    args = parser.parse_args()
    
    args = vars(args)
    
    main(args)
