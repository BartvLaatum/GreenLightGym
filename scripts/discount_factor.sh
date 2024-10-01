#!/bin/bash

# Change to the directory containing the Python script
cd ./GreenLightGym/

# Run the Python script
python -m greenlight_gym.experiments.train_agent --env_id GreenLightHeatCO2 --project discount-factor --group gamma-0999 --algorithm ppo-99 --env_config_name 5min_time --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
python -m greenlight_gym.experiments.train_agent --env_id GreenLightHeatCO2 --project discount-factor --group gamma-0998 --algorithm ppo-98 --env_config_name 5min_time --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
python -m greenlight_gym.experiments.train_agent --env_id GreenLightHeatCO2 --project discount-factor --group gamma-0997 --algorithm ppo-97 --env_config_name 5min_time --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
python -m greenlight_gym.experiments.train_agent --env_id GreenLightHeatCO2 --project discount-factor --group gamma-0996 --algorithm ppo-96 --env_config_name 5min_time --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
python -m greenlight_gym.experiments.train_agent --env_id GreenLightHeatCO2 --project discount-factor --group gamma-0995 --algorithm ppo-95 --env_config_name 5min_time --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
python -m greenlight_gym.experiments.train_agent --env_id GreenLightHeatCO2 --project discount-factor --group gamma-0994 --algorithm ppo-94 --env_config_name 5min_time --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
