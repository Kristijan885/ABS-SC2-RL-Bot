from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from sc2env import PySC2GymWrapper
from absl import app


# Define the main function
def main(argv):
    env = PySC2GymWrapper([9, 84, 84], visualize=True)

    # Separate env for evaluation
    # eval_env = PySC2GymWrapper(5)

    model = PPO('MlpPolicy', env, verbose=1)    #, tensorboard_log="logs/")

    # Random Agent, before training
    # mean_reward, std_reward = evaluate_policy(
    #     model,
    #     eval_env,
    #     n_eval_episodes=10,
    #     deterministic=True,
    # )
    #
    # print(f"Random actions: mean_reward={mean_reward:.2f} +/- {std_reward}")

    # Train the model
    model.learn(total_timesteps=10000)

    model.save('./models/minigames/buildMarines.zip')

    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    # print(f"Trained model: mean_reward={mean_reward:.2f} +/- {std_reward}")


# Ensure that the script runs through absl.app to parse FLAGS
if __name__ == '__main__':
    app.run(main)
