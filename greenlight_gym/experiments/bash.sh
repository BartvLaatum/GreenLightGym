#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Run the Python script
# python experiments/train_agent.py --env_id GreenLightHeatCO2
python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-kco2-1e-4 --env_config_name four_controls --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-kco2-1e-4 --env_config_name four_controls --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 667 --n_evals 10
python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-kco2-1e-4 --env_config_name four_controls --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 668 --n_evals 10
python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-kco2-1e-4 --env_config_name four_controls --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 669 --n_evals 10
python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-kco2-1e-4 --env_config_name four_controls --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 670 --n_evals 10
