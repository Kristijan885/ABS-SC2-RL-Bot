from stable_baselines3 import PPO

from sc2env import PySC2GymWrapper


if __name__ == '__main__':
    env = PySC2GymWrapper(6, visualize=True)

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="logs/test")

    model.learn(total_timesteps=10000)