#!/bin/bash

# Change to the directory containing the Python script
cd ./GreenLightGym/

python -m greenlight_gym.experiments.omega_pen_sweep --env_id GreenLightHeatCO2 --project multiplicative-penalty --group omega --algorithm ppo-98 --env_config_name multiplicative_pen --start_range 0.1 --end_range 1.0 --n_values 10 --total_timesteps 10_000_000 --n_eval_episodes 60 --SEED 666 --n_evals 10
