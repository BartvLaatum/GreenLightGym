#!/bin/bash

# Change to the directory containing the Python script
cd ./GreenLightGym/

# Run the Python script
for SEED in 666 667 668 669 670
do
    python -m greenlight_gym/experiments/train_agent --env_id GreenLightHeatCO2 --project benchmark --group additive-0.99 --env_config_name additive_pen_daily_avg_temp --algorithm ppo-99 --total_timesteps 40_000_000 --n_eval_episodes 60 --env_seed 666 --num_cpus 12 --model_seed ${SEED} --n_evals 40 --save_model --save_env
done
