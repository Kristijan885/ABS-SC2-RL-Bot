import os
import tempfile

import numpy as np
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from dagger_expert_policy import ExpertPolicy
from dagger_demo_monkey_patch import safe_save_dagger_demo
import imitation.algorithms.dagger

imitation.algorithms.dagger._save_dagger_demo = safe_save_dagger_demo


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
        learning_rate=1e-3,  # Higher learning rate
        n_steps=512,  # Smaller steps
        batch_size=32,  # Smaller batch size
        n_epochs=20,  # More epochs
        gamma=0.99
    )

    bc_trainer = BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=ppo_model.policy,
        rng=np.random.default_rng(42),
        batch_size=32,
        l2_weight=0.001,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="dagger_") as tmpdir:
            print(f"Using temporary directory: {tmpdir}")

            dagger_trainer = SimpleDAggerTrainer(
                venv=venv,
                scratch_dir=tmpdir,
                expert_policy=ExpertPolicy(env),
                bc_trainer=bc_trainer,
                beta_schedule=lambda step: 0.9,  # High expert reliance
                rng=np.random.default_rng(42),
            )

            print("Starting DAgger training...")
            try:
                dagger_trainer.train(
                    total_timesteps=144000,
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
