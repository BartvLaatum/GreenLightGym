#!/bin/bash

# Change to the directory containing the Python script
cd ./GreenLightGym/

# Run the Python script
python -m greenlight_gym/experiments/train_agent.py --env_id GreenLightHeatCO2 --project effect-daily-avg-temp --group additive-penalty-prev-controls --env_config_name additive_previous_control --algorithm ppo-98 --total_timesteps 20_000_000 --n_eval_episodes 60 --seed 666 --n_evals 20 --save_model --save_env
python -m greenlight_gym/experiments/train_agent.py --env_id GreenLightHeatCO2 --project effect-daily-avg-temp --group multiplicative-penalty-prev-controls --env_config_name multiplicative_previous_control --algorithm ppo-98 --total_timesteps 20_000_000 --n_eval_episodes 60 --seed 666 --n_evals 20 --save_model --save_env
