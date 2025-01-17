import os
import tempfile

import numpy as np
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from dagger_expert_policy import ExpertPolicy


def dagger_training(env, model_path="./models/dagger_model.zip"):
    """
    Pre-train a PPO model using DAgger, preparing it for further fine-tuning.

    Args:
        env: The environment to train in
        model_path: Path to save the trained model

    Returns:
        PPO: The trained PPO model
    """
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Properly wrap the environment
    if isinstance(env, DummyVecEnv):
        venv = env
    else:
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

    ppo_model = PPO(
        "CnnPolicy",
        venv,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={"net_arch": [dict(pi=[64, 64], vf=[64, 64])]}
    )

    bc_trainer = BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=ppo_model.policy,
        rng=np.random.default_rng(42),
        batch_size=32,
        l2_weight=0.01
    )

    try:
        with tempfile.TemporaryDirectory(prefix="dagger_") as tmpdir:
            print(f"Using temporary directory: {tmpdir}")

            dagger_trainer = SimpleDAggerTrainer(
                venv=venv,
                scratch_dir=tmpdir,
                expert_policy=ExpertPolicy(env),
                bc_trainer=bc_trainer,
                rng=np.random.default_rng(42),
            )

            print("Starting DAgger training...")
            try:
                dagger_trainer.train(
                    total_timesteps=200000,
                )
            except Exception as e:
                print(f"Error during DAgger training: {str(e)}")
                raise

            print(f"Saving model to {model_path}")
            ppo_model.save(model_path)
            print("Model saved successfully")

            return ppo_model

    except Exception as e:
        print(f"Error in DAgger training: {str(e)}")
        raise
