#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Run the Python script

for SEED in 669 670
do
    python experiments/train_agent.py --env_id GreenLightHeatCO2 --project benchmark --group additive-0.99 --env_config_name additive_pen_daily_avg_temp --algorithm ppo-99 --total_timesteps 40_000_000 --n_eval_episodes 60 --env_seed 666 --num_cpus 12 --model_seed ${SEED} --n_evals 40 --save_model --save_env
done
