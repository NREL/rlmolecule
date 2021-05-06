from rlmolecule.gym.alphazero_gym import AlphaZeroGymEnv

class TestGym(AlphaZeroGymEnv):
    def get_obs(self):
        return self.env.get_obs()
