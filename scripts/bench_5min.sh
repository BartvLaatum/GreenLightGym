#!/bin/bash

# Change to the directory containing the Python script
cd ./GreenLightGym/

# Run the Python script
# python experiments/train_agent.py --env_id GreenLightHeatCO2
python -m greenlight_gym/experiments/train_agent --env_id GreenLightHeatCO2 --project test-max-pen-ranges --group benchmark-5min-stricter --algorithm ppo --env_config_name 5min_four_controls --total_timesteps 1_000_000 --n_eval_episodes 60 --seed 666 --n_evals 2
