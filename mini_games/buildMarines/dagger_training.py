from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from imitation.algorithms.bc import BC
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.policies.base import RandomPolicy

from dagger_expert_policy import ExpertPolicy

from stable_baselines3 import PPO
import numpy as np


def dagger_training(env, model_path="./models/dagger_model.zip"):
    """
    Pre-train a PPO model using DAgger, preparing it for Optuna fine-tuning.
    The model will maintain its architecture and parameters for further optimization.
    """

    # Wrap the environment in a DummyVecEnv if needed
    if not hasattr(env, 'num_envs'):
        venv = DummyVecEnv([lambda: env])

    # Initialize the PPO model that will be trained
    ppo_model = PPO("CnnPolicy", env, verbose=1)

    # Create scratch directory if it doesn't exist
    scratch_dir = "./dagger_scratch"

    # Create BC trainer that will update the PPO model's policy
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=None,
        policy=ppo_model.policy,  # Use the PPO's policy directly
        device="auto",
        rng=np.random.seed(seed=48)
    )

    # Initialize DAgger trainer
    dagger_trainer = SimpleDAggerTrainer(
        venv=venv,
        bc_trainer=bc_trainer,
        expert_policy=ExpertPolicy(env),
        scratch_dir=scratch_dir,
        beta_schedule=lambda step: max(0.9 - step * 0.1, 0),
        rng=np.random.default_rng(seed=42)
    )

    # Run DAgger iterations
    for i in range(5):
        print(f"Starting DAgger iteration {i + 1}/5")
        dagger_trainer.train(2000)

    # Save the pre-trained PPO model
    ppo_model.save(model_path)
    print(f"DAgger pre-trained PPO model saved to {model_path}")

    return ppo_model
