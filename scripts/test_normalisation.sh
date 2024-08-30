#!/bin/bash

# Change to the directory containing the Python script
cd ./GreenLightGym/

python -m greenlight_gym/experiments/train_agent --env_id GreenLightHeatCO2 --project benchmark-ppo --group no-reward-normalisation --algorithm ppo --env_config_name 5min_time --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 25
