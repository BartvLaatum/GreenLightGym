#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Run the Python script
# python experiments/train_agent.py --env_id GreenLightHeatCO2


python experiments/omega_pen_sweep.py --env_id GreenLightHeatCO2 --project multiplicative-penalty --group omega --algorithm ppo-98 --env_config_name multiplicative_pen --start_range 0.1 --end_range 1.0 --n_values 10 --total_timesteps 10_000_000 --n_eval_episodes 60 --SEED 666 --n_evals 10
