#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Run the Python script

for SEED in 666 667 668 669 670
do
    python experiments/train_agent.py --env_id GreenLightHeatCO2 --project convergence-test --group multiplicative-0.99 --env_config_name multiplicative_pen_daily_avg_temp --algorithm ppo-99 --total_timesteps 10_000_000 --n_eval_episodes 60 --env_seed 666 --model_seed ${SEED} --n_evals 10 --save_model --save_env
done