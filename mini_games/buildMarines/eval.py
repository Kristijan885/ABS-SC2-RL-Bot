import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from absl import app, flags

from mini_games.buildMarines.build_marines_actions import BuildMarinesActionManager
# Import your custom SC2 environment
from sc2env import sc2_env, PySC2GymWrapper


def evaluate_model(_):
    """
    Evaluates the given model in the provided environment.

    Args:
        model_path (str): Path to the trained SB3 model.
        env_fn (callable): Function that returns a new environment instance.
        num_episodes (int): Number of episodes to run for evaluation.

    Returns:
        float: The average reward over the episodes.
    """
    model_path = "./models/dagger_model_bc.zip"
    num_episodes = 10

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load the trained model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    env = PySC2GymWrapper(num_actions=[5, 84, 84], action_manager=BuildMarinesActionManager(), step_mul=None)

    # Initialize metrics
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Predict the action using the loaded model
            action = model.predict(obs, deterministic=True)
            print(f"action : {action[0]} | full action: {action}")

            # Take a step in the environment
            obs, reward, done, info, _ = env.step(action[0])

            # Accumulate rewards
            episode_reward += reward

            if done:
                print(f"Episode {episode + 1} finished with reward: {episode_reward}")
                total_rewards.append(episode_reward)

    # Calculate average reward
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    # Close the environment
    env.close()

    return avg_reward


if __name__ == "__main__":

    # Evaluate the model
    app.run(evaluate_model)
