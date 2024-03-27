#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Run the Python script
# python experiments/train_agent.py --env_id GreenLightHeatCO2

# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project benchmark-ppo --group kco2-4e-4 --algorithm ppo --env_config_name 5min_four_controls --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 50
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project benchmark-ppo --group pred-hor --algorithm ppo --env_config_name 5min_pred_hor --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 50
# python experiments/train_agent.py --/env_id GreenLightHeatCO2 --project benchmark-ppo --group cyclic-time --algorithm ppo --env_config_name 5min_time --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 50

python experiments/train_agent.py --env_id GreenLightHeatCO2 --project multiplicative-penalty --group k_hum-0.4 --algorithm ppo-98 --env_config_name multiplicative_pen --total_timesteps 10_000_000 --n_eval_episodes 60 --seed 666 --n_evals 10
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project benchmark-ppo --group cyclic-time-H-5hr-larger-nets-lnlr-diff-rew-coeffs --algorithm ppo-99 --env_config_name 5min_time_hor_5hr --total_timesteps 20_000_000 --n_eval_episodes 60 --seed 666 --n_evals 4
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project benchmark-ppo --group cyclic-time-H-1D-high-gamma-gae-0995-larger-nets --algorithm ppo-99 --env_config_name 5min_time_pred_hor --total_timesteps 20_000_000 --n_eval_episodes 60 --seed 666 --n_evals 1
