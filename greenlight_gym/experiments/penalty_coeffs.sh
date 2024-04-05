#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

for SEED in 666
do
    python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project multiplicative-penalty-2 --group co2 --algorithm ppo-98 --env_config_name 5min_time --start_range 0 --end_range 1e-2 --n_values 6  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 10_000_000 --SEED $SEED
    python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project multiplicative-penalty-2 --group temp --algorithm ppo-98 --env_config_name 5min_time --start_range 0 --end_range 2 --n_values 6  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 10_000_000 --SEED $SEED
    python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project multiplicative-penalty-2 --group rh --algorithm ppo-98 --env_config_name 5min_time --start_range 0 --end_range 1 --n_values 6  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 10_000_000 --SEED $SEED
done
