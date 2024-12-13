from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from build_marines_actions import BuildMarinesActionManager
from eval_random_agent import evaluate
from sc2env import PySC2GymWrapper
from absl import app, flags
import optuna
import json


def objective(trial):
    """Objective function for Optuna to optimize PPO hyperparameters."""
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.9999)
    n_steps = trial.suggest_int('n_steps', 128, 2048, step=128)

    env = PySC2GymWrapper(num_actions=[6, 84, 84], action_manager=BuildMarinesActionManager())
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        n_steps=n_steps,
    )

    model.learn(total_timesteps=144000)

    mean_reward, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=5,
        deterministic=True,
    )

    env.close()

    return mean_reward


def main(_):
    if flags.FLAGS.eval_random:
        evaluate()

    study = optuna.create_study(direction='maximize', study_name='ABS-SC2-BuildMarines')
    study.optimize(objective, n_trials=10)

    print("\nBest Hyperparameters:")
    print(study.best_params)
    with open("optuna_study.json", "w") as f:
        json.dump(study.trials_dataframe().to_dict(orient="list"), f)

    best_params = study.best_params
    env = PySC2GymWrapper(num_actions=[6, 84, 84], action_manager=BuildMarinesActionManager())
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=best_params['learning_rate'],
        gamma=best_params['gamma'],
        ent_coef=best_params['ent_coef'],
        n_steps=best_params['n_steps'],
    )

    model_path = './models/buildMarines_optuna.zip'
    # One episode is 14400 steps
    model.learn(total_timesteps=1440000)
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
