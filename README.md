# SC2-BuildMarines-RL

A StarCraft II bot I built that learns to crank out marines using a mix of imitation learning and reinforcement learning techniques.

## What it does

This bot learns to efficiently run a marine production line in the SC2 BuildMarines mini-game:
- Selects SCVs to do the dirty work
- Builds supply depots right before hitting the supply cap
- Places barracks in good spots
- Pumps out marines as fast as resources allow

## How it works

- Built with PySC2 (the official StarCraft II Python API)
- Uses PPO from Stable-Baselines3 for the reinforcement learning magic
- Implements DAgger (a cool imitation learning technique) to learn from my hand-coded expert
- Uses Optuna to find the best hyperparameters without me having to guess

## Code structure

- `build_marines_actions.py` - Defines all possible game actions
- `dagger_expert_policy.py` - My hand-coded expert that knows the basics
- `dagger_training.py` - DAgger implementation to learn from the expert
- `train_build_marines.py` - Main training script with hyperparameter tuning
- `eval.py` - Scripts to test how good the bot actually is

## Getting started

1. Clone this repo
2. Make sure you have StarCraft II installed with the mini-games
3. Install requirements
4. Run training or evaluation scripts (see docs folder)

## Requirements

- Python 3.7+
- StarCraft II + Mini-game maps
- PySC2
- Stable-Baselines3
- Optuna
- Imitation

## License

MIT
