from mini_games.buildMarines.build_marines_actions import BuildMarinesActionManager
from sc2env import PySC2GymWrapper
import numpy as np


def evaluate(n_eval_episodes=10):
    """Evaluate a random agent by sampling actions from the action space."""
    print("Evaluating random actions agent...")
    env = PySC2GymWrapper(num_actions=[6, 84, 84], action_manager=BuildMarinesActionManager(), visualize=False)

    total_rewards = []

    for episode in range(n_eval_episodes):
        _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Sample a random action from the environment's action space
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    env.close()

    print(f"Random agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
