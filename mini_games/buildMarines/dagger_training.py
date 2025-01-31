import os
import tempfile

import numpy as np
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
        # infos.append(info)

        obs = next_obs

    # Convert lists to NumPy arrays
    # for Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [4, 1, 3, 168, 168] (batch dim)
    states = np.array(states).squeeze()
    actions = np.array(actions)
    rewards = np.array(rewards).squeeze()
    # infos = np.array(infos, dtype=object)  # Use dtype=object for complex info objects

    # Include the terminal field
    terminal = done
    return TrajectoryWithRew(obs=states, acts=actions, rews=rewards, infos=None, terminal=terminal)


def dagger_training(env, model_path="./models/dagger_model.zip"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Wrap environment
    if not isinstance(env, DummyVecEnv):
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])  # Wrap for Stable-Baselines3
        venv = VecTransposeImage(venv)  # Converts channel-last to channel-first
    else:
        venv = env

    # venv.envs[0].env
    print(env.observation_space)
    print(venv.observation_space)

    # Initialize PPO with smaller networks for overfitting
    ppo_model = PPO(
        "CnnPolicy",
        venv,
        verbose=1,
        learning_rate=5e-4,
        n_steps=128,
        batch_size=8,  # Very small batch size for overfitting
        n_epochs=100,  # More epochs
        ent_coef=0.0,  # Disable entropy to encourage overfitting
        clip_range=0.1,  # Smaller clip range
        gamma=0.99,
    )

    print(ppo_model.policy.observation_space)

    # Collect single demonstration
    expert_policy = ExpertPolicy(venv)
    demo = collect_single_demonstration(venv, expert_policy)
    expert_data = flatten_trajectories([demo])

    # Configure BC for extreme overfitting
    bc_trainer = BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=ppo_model.policy,
        rng=np.random.default_rng(42),
        batch_size=4,  # Tiny batch size
        demonstrations=expert_data,
        l2_weight=0.0,  # No regularization
    )

    print("Training BC...")
    bc_trainer.train(n_epochs=3000)  # More epochs for overfitting

    # Save BC model
    bc_model_path = model_path.replace(".zip", "_bc.zip")
    ppo_model.save(bc_model_path)

    # try:
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         dagger_trainer = SimpleDAggerTrainer(
    #             venv=venv,
    #             scratch_dir=tmpdir,
    #             expert_policy=expert_policy,
    #             bc_trainer=bc_trainer,
    #             beta_schedule=lambda step: 0.8,  # High reliance on expert
    #             rng=np.random.default_rng(42),
    #         )
    #
    #         dagger_trainer.extend_and_update([demo])
    #         print("Training DAgger...")
    #         dagger_trainer.train(total_timesteps=5000)
    #
    #         ppo_model.save(model_path)
    #         return ppo_model
    #
    # except Exception as e:
    #     print(f"DAgger training error: {str(e)}")
    #     raise

