#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Run the Python script

for SEED in 666
do
    python experiments/train_agent.py --env_id GreenLightHeatCO2 --project eval_training --group multiplicative-0.99 --env_config_name train_eval_set --algorithm ppo-99 --total_timesteps 40_000_000 --n_eval_episodes 60 --env_seed 666 --num_cpus 12 --model_seed ${SEED} --n_evals 40 --save_model --save_env
done
