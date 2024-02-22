#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Run the Python script
# python experiments/train_agent.py --env_id GreenLightHeatCO2
python experiments/train_agent.py --env_id GreenLightHeatCO2 --project test-max-pen-ranges --group benchmark-5min-stricter --algorithm ppo --env_config_name 5min_four_controls --total_timesteps 1_000_000 --n_eval_episodes 60 --seed 666 --n_evals 2
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-pred-hor-5min-kco2-3e-4 --algorithm ppo --env_config_name four_controls_pred_hor_5min --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-pred-hor-0min-gamma-0.98-kco2-3e-4 --algorithm ppo_gamma_0.98 --env_config_name four_controls_pred_hor_0min --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-pred-hor-5min-gamma-0.98-kco2-3e-4 --algorithm ppo_gamma_0.98 --env_config_name four_controls_pred_hor_5min --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-pred-hor-0min-gamma-0.98_lr3e-6-kco2-3e-4 --algorithm ppo_gamma_0.98_lr3e-6 --env_config_name four_controls_pred_hor_0min --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 666 --n_evals 1
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-pred-hor-0min-gamma-0.98kco2-1e-3 --algorithm ppo_gamma_0.98 --env_config_name four_controls_pred_hor_0min --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 666 --n_evals 1
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multi-eval --group benchmark-5min-pred-hor-5min-gamma-0.98kco2-1e-3 --algorithm ppo_gamma_0.98 --env_config_name four_controls_pred_hor_5min --total_timesteps 2_000_000 --n_eval_episodes 60 --seed 666 --n_evals 1
