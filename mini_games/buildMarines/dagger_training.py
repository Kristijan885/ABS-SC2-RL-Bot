import os
import tempfile

import numpy as np
from absl import app, flags
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.rollout import flatten_trajectories, make_sample_until
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from dagger_expert_policy import ExpertPolicy
from dagger_demo_monkey_patch import safe_save_dagger_demo
import imitation.algorithms.dagger

from mini_games.buildMarines.build_marines_actions import BuildMarinesActionManager
from sc2env import PySC2GymWrapper

imitation.algorithms.dagger._save_dagger_demo = safe_save_dagger_demo


def collect_single_demonstration(venv, expert_policy):
    obs = venv.reset()
    done = False
    states, actions, rewards, infos = [obs], [], [], []

    while not done:
        action, _ = expert_policy.predict(obs)
        next_obs, reward, done, _ = venv.step(action)
        states.append(next_obs)
        actions.append(action[0])
        rewards.append(reward)

        obs = next_obs

    # Squeeze to 4D (batched) tensor
    states = np.array(states).squeeze()
    actions = np.array(actions)
    rewards = np.array(rewards).squeeze()

    terminal = done
    return TrajectoryWithRew(obs=states, acts=actions, rews=rewards, infos=None, terminal=terminal)


def dagger_training(env, model_path="./models/dagger_model.zip"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not isinstance(env, DummyVecEnv):
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
        venv = VecTransposeImage(venv)  # Converts channel-last to channel-first
    else:
        venv = env

    print("Observation Space (Env):", env.observation_space)
    print("Observation Space (VecEnv):", venv.observation_space)

    if flags.FLAGS.load_model:
        ppo_model = PPO.load(model_path.replace(".zip", f"_{flags.FLAGS.load_model}.zip"))
    else:
        ppo_model = PPO(
            "CnnPolicy",
            venv,
            verbose=1,
            learning_rate=3e-4,  # Slightly reduced for stability
            n_steps=512,  # Longer rollout to stabilize learning
            batch_size=64,  # Larger batch size to improve generalization
            n_epochs=10,  # Fewer epochs to prevent overfitting
            ent_coef=0.01,  # Encourage exploration
            clip_range=0.2,  # More stable training
            gamma=0.99,
        )

    print("PPO Policy Observation Space:", ppo_model.policy.observation_space)

    expert_policy = ExpertPolicy(venv)

    should_train_bc = flags.FLAGS.train_bc

    if should_train_bc:
        demo = collect_single_demonstration(venv, expert_policy)
        expert_data = flatten_trajectories([demo])
        print("Finished collecting single demonstration for BC expert data.")

    bc_trainer = BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=ppo_model.policy,
        rng=np.random.default_rng(42),
        batch_size=64,  # Larger batch size to generalize better
        demonstrations=expert_data if should_train_bc else None,
        l2_weight=1e-4,  # Add small regularization to prevent overfitting
    )

    if should_train_bc:
        print("Training BC...")
        bc_trainer.train(n_epochs=500)

        bc_model_path = model_path.replace(".zip", "_bc.zip")
        ppo_model.save(bc_model_path)

    for i in range(flags.FLAGS.dagger_range, 100):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                dagger_trainer = SimpleDAggerTrainer(
                    venv=venv,
                    scratch_dir=tmpdir,
                    expert_policy=expert_policy,
                    bc_trainer=bc_trainer,
                    beta_schedule=lambda step: max(0.1, 1.0 - step / 100000),  # Gradually reduce expert reliance
                    rng=np.random.default_rng(42),
                )

                print("Training DAgger...")
                dagger_trainer.train(total_timesteps=15000)  # Lower timesteps because of my memory limitations

                if i % 5 == 0:
                    model_path = model_path.replace(".zip", f"_{i}.zip")
                    print(f"Saving model {model_path}")
                    ppo_model.save(model_path)

        except Exception as e:
            print(f"DAgger training error: {str(e)}")
            raise

    return ppo_model


def main(_):
    env = PySC2GymWrapper(num_actions=[6, 84, 84], action_manager=BuildMarinesActionManager(), step_mul=None)

    dagger_training(env)


if __name__ == '__main__':
    flags.DEFINE_integer('dagger_range', 46, 'Number of times to restart DAgger training (for reduced memory usage for demonstrations)')
    flags.DEFINE_integer('load_model', 45, 'Load previously trained model for further training (format dagger_model_{N}.zip)')
    flags.DEFINE_boolean('train_bc', False, 'Whether or not to train BC with single demonstration for eval')

    app.run(main)
