from stable_baselines3 import PPO
from sc2env import PySC2GymWrapper
from absl import app

# Define the main function
def main(argv):
    # Initialize your custom PySC2 environment
    env = PySC2GymWrapper(6, visualize=True)

    # Create and configure the PPO model
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="logs/test")

    # Train the model
    model.learn(total_timesteps=10000)

# Ensure that the script runs through absl.app to parse FLAGS
if __name__ == '__main__':
    app.run(main)
