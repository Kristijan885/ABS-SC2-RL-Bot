import os
import pickle
import optuna
from absl import app, flags
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from build_marines_actions import BuildMarinesActionManager
from eval_random_agent import evaluate
from sc2env import PySC2GymWrapper


def create_new_model(env, trial):
    """Create a new PPO model with trial parameters while maintaining original architecture."""
    model = PPO(
        policy='CnnPolicy',
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
        clip_range=0.2,
        gamma=0.99,
    )

    pretrained_path = 'models/dagger_model_55.zip'
    model.load(pretrained_path)

    model.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    model.ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
    model.gamma = trial.suggest_float('gamma', 0.9, 0.9999)

    return model


def objective(trial):
    """Objective function for Optuna to optimize PPO hyperparameters."""
    try:
        env = PySC2GymWrapper(
            num_actions=[6, 84, 84],
            action_manager=BuildMarinesActionManager(),
            step_mul=None
        )

        model = create_new_model(env, trial)

        log_dir = f"./tensorboard_logs/trial_{trial.number}"
        os.makedirs(log_dir, exist_ok=True)
        logger = configure(log_dir, ["stdout", "tensorboard"])
        model.set_logger(logger)

        model.learn(total_timesteps=35000)

        print(f"Evaluating trial {trial.number}:")
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=10,
            deterministic=True
        )

        model_path = f'./models/trial_{trial.number}.zip'
        model.save(model_path)
        print(f"Model with mean reward {mean_reward:.2f} ± {std_reward:.2f} saved to {model_path}")

        return mean_reward

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        raise e

    finally:
        env.close()


def main(_):
    if flags.FLAGS.eval_random:
        evaluate()

    study = optuna.create_study(
        direction='maximize',
        study_name='ABS-SC2-BuildMarines',
        pruner=optuna.pruners.MedianPruner()
    )

    try:
        study.optimize(objective, n_trials=20)

        print("\nBest Hyperparameters:")
        print(study.best_params)

        study_path = './optuna_study.pkl'
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        print(f"Optuna study saved to {study_path}")

        env = PySC2GymWrapper(
            num_actions=[5, 84, 84],
            action_manager=BuildMarinesActionManager(),
            step_mul=None
        )

        model = PPO(
            policy='CnnPolicy',
            env=env,
            verbose=1,
            learning_rate=study.best_params['learning_rate'],
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            ent_coef=study.best_params['ent_coef'],
            clip_range=0.2,
            gamma=study.best_params['gamma']
        )

        model.load('models/dagger_model_55.zip')

        log_dir = "./tensorboard_logs/final_training"
        os.makedirs(log_dir, exist_ok=True)
        logger = configure(log_dir, ["stdout", "tensorboard"])
        model.set_logger(logger)

        model.learn(total_timesteps=1440000)  # One episode is 14400 steps

        model_path = './models/buildMarines_optuna.zip'
        model.save(model_path)
        print(f"Model saved to {model_path}")

        mean_reward_trained, std_reward_trained = evaluate_policy(
            model,
            env,
            n_eval_episodes=10,
            deterministic=True
        )
        print(f"Trained agent (optimized): mean_reward={mean_reward_trained:.2f} +/- {std_reward_trained:.2f}")

    finally:
        env.close()


if __name__ == '__main__':
    flags.DEFINE_boolean('eval_random', False, 'Whether or not to evaluate a random action agent first')
    app.run(main)