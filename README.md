# SC2-BuildMarines-RL

A StarCraft II bot we built that learns to crank out marines using a mix of imitation learning and reinforcement learning techniques.

## What it does

This bot learns to efficiently produce marines in the SC2 BuildMarines mini-game by performing these key actions:
- Selecting worker SCVs
- Selecting barracks buildings
- Building supply depots to increase population capacity
- Constructing barracks for marine production
- Training marines whenever resources are available
- Managing resource allocation between buildings and units

## Environment

The environment is wrapped with a custom Gym interface that:
- Uses a MultiDiscrete action space [6, 84, 84] representing:
 - Action type (select SCV, build barracks, build supply, etc.)
 - X coordinate on the map
 - Y coordinate on the map
- Processes raw SC2 observations into a format suitable for CNN-based policies
- Handles reward calculation based on marine production and resource efficiency
- Manages transitions between game states and episode termination

## How it works

- Built with PySC2 (the official StarCraft II Python API)
- Uses PPO from Stable-Baselines3 for the reinforcement learning magic
- Implements DAgger (a cool imitation learning technique) to learn from my hand-coded expert
- Uses Optuna to find the best hyperparameters without me having to guess

## Code structure

- `build_marines_actions.py` - Defines all possible game actions
- `dagger_expert_policy.py` - Our hand-coded expert that knows the basics
- `dagger_training.py` - DAgger implementation to learn from the expert
- `train_build_marines.py` - Main training script with hyperparameter tuning
- `eval.py` - Scripts to test how good the bot actually is

## Getting started

1. Clone this repo
2. Make sure you have StarCraft II installed with the mini-games
3. Install requirements
4. Run training or evaluation scripts

## Requirements

- Python 3.7+
- StarCraft II + Mini-game maps (downloaded seperately)
- PySC2
- Stable-Baselines3
- Optuna
- Imitation

## License

MIT

## References and Research

This project was evaluated by us in [Reinforcement Learning Methodologies for DeepMind's StarCraft-2 Mini-game Environment](DAgger%20Imitation%20Learning%20StarCraft%202.pdf) (Krstev, Panchevski, Gievska & Tosheska, 2025). (Krstev, Panchevski, Gievska & Tosheska, 2025).
