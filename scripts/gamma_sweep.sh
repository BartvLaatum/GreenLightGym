#!/bin/bash

# Change to the directory containing the Python script
cd ./GreenLightGym/

# Run the Python script
python -m greenlight_gym/experiments/gamma_sweep --env_id GreenLightHeatCO2 --project gamma-sweep --group multiplicative --algorithm ppo-98 --env_config_name multiplicative_pen_daily_avg_temp --start_range 0.95 --end_range 0.995 --n_values 10 --total_timesteps 10_000_000 --n_eval_episodes 60 --SEED 666 --n_evals 10
python -m greenlight_gym/experiments/gamma_sweep --env_id GreenLightHeatCO2 --project gamma-sweep --group additive --algorithm ppo-98 --env_config_name additive_pen_daily_avg_temp --start_range 0.95 --end_range 0.995 --n_values 10 --total_timesteps 10_000_000 --n_eval_episodes 60 --SEED 666 --n_evals 10
