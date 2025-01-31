from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from build_marines_actions import BuildMarinesActionManager
from dagger_training import dagger_training
from eval_random_agent import evaluate
from sc2env import PySC2GymWrapper
from absl import app, flags
import optuna
import pickle
import numpy as np


def objective(trial):
    """Objective function for Optuna to optimize PPO hyperparameters."""
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 0.5, 0.9999)
    n_steps = trial.suggest_int('n_steps', 128, 2048, step=128)
    env = PySC2GymWrapper(num_actions=[6, 84, 84], action_manager=BuildMarinesActionManager(), step_mul=None)

    # Better practice is to load weights instead of entire model
    # Load the pretrained DAgger model
    dagger_model_path = './models/dagger_model.zip'
    model = PPO.load(dagger_model_path, env=env)

    # Update the model with trial's hyperparameters
    model.learning_rate = learning_rate
    model.gamma = gamma
    model.ent_coef = ent_coef
    model.n_steps = n_steps

    model.learn(total_timesteps=1000000)

    print(f"Evaluating trial {trial.number}:")

    mean_reward, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )

    model_path = f'./models/trial_{trial.number}.zip'
    model.save(model_path)
    print(f"Model with mean reward {mean_reward} saved to {model_path}")

    env.close()

    return mean_reward


def main(_):
    if flags.FLAGS.eval_random:
        evaluate()

    # Limit memory usage for numpy
    np.ones(1).nbytes  # Force numpy to initialize
    np.set_printoptions(threshold=1000)

    env = PySC2GymWrapper(num_actions=[6, 84, 84], action_manager=BuildMarinesActionManager(), step_mul=None)

    # Step 1: Train with DAgger
    dagger_model_path = './models/dagger_model.zip'
    dagger_model = dagger_training(env, model_path=dagger_model_path)

    study = optuna.create_study(direction='maximize', study_name='ABS-SC2-BuildMarines')
    study.optimize(objective, n_trials=10)

    print("\nBest Hyperparameters:")
    print(study.best_params)

    study_path = './optuna_study.pkl'
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"Optuna study saved to {study_path}")

    best_trial = study.best_trial

    best_model_path = f'./models/ppo_trial_{best_trial.number}.zip'
    env = PySC2GymWrapper(num_actions=[5, 84, 84], action_manager=BuildMarinesActionManager(), step_mul=None)
    model = PPO.load(best_model_path, env=env)
    print(f"Loaded model from {best_model_path} for additional training")

    # One episode is 14400 steps
    model.learn(total_timesteps=144000)

    model_path = './models/buildMarines_optuna.zip'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    mean_reward_trained, std_reward_trained = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )
    print(f"Trained agent (optimized): mean_reward={mean_reward_trained:.2f} +/- {std_reward_trained:.2f}")

    env.close()


if __name__ == '__main__':
    flags.DEFINE_boolean('eval_random', False, 'Whether or not to evaluate a random action agent first')

    app.run(main)
