import os

import numpy as np
from absl import app, flags
from stable_baselines3 import PPO

from mini_games.buildMarines.build_marines_actions import BuildMarinesActionManager
from sc2env import PySC2GymWrapper


def evaluate_model(model_path, num_episodes=50):
    """
    Evaluates the given model in the provided environment.

    Args:
        model_path (str): Path to the trained SB3 model.
        num_episodes (int): Number of episodes to run for evaluation.

    Returns:
        float: The average reward over the episodes.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    env = PySC2GymWrapper(num_actions=[5, 84, 84], action_manager=BuildMarinesActionManager(), step_mul=None)

    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = model.predict(obs, deterministic=True)
            obs, reward, done, info, _ = env.step(action[0])

            episode_reward += reward

            if done:
                print(f"Episode {episode + 1} finished with reward: {episode_reward}")
                total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    env.close()

    return avg_reward


def main(_):
    evaluate_model(flags.FLAGS.model_path)


if __name__ == "__main__":
    flags.DEFINE_string('model_path', './models/dagger_model_23.zip', 'Path to the pretrained model')
    app.run(main)
