from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from build_marines_actions import BuildMarinesActionManager
from sc2env import PySC2GymWrapper
import numpy as np
from absl import app


def evaluate_random_agent(env, n_eval_episodes=10):
    """Evaluate a random agent by sampling actions from the action space."""
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
    return mean_reward, std_reward


def main(_):
    # eval_env = PySC2GymWrapper(num_actions=[6, 84, 84], action_manager=BuildMarinesActionManager(), visualize=False)
    #
    # print("Evaluating random actions agent...")
    # mean_reward_random, std_reward_random = evaluate_random_agent(eval_env, n_eval_episodes=10)
    # print(f"Random agent: mean_reward={mean_reward_random:.2f} +/- {std_reward_random:.2f}")
    #
    # eval_env.close()

    env = PySC2GymWrapper(num_actions=[6, 84, 84], action_manager=BuildMarinesActionManager())
    model = PPO('MlpPolicy', env, verbose=1)

    model_path = './models/buildMarines.zip'
    model.load(model_path, env)

    print("Training the PPO model...")
    model.learn(total_timesteps=10000)

    model.save(model_path)
    print(f"Model saved to {model_path}")

    print("Evaluating trained model...")

    mean_reward_trained, std_reward_trained = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )

    # mean_reward_trained, std_reward_trained = evaluate_trained_model(env, model, n_eval_episodes=10)
    print(f"Trained agent: mean_reward={mean_reward_trained:.2f} +/- {std_reward_trained:.2f}")

    print("\nComparison:")
    # print(f"Random agent: mean_reward={mean_reward_random:.2f} +/- {std_reward_random:.2f}")
    print(f"Trained agent: mean_reward={mean_reward_trained:.2f} +/- {std_reward_trained:.2f}")

    env.close()


# Ensure that the script runs through absl.app to parse FLAGS
if __name__ == '__main__':
    app.run(main)
