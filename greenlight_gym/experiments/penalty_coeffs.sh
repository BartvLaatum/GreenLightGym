#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

for SEED in 666
do
    python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project additive-penalty --group co2 --algorithm ppo --start_range 0 --end_range 1e-2 --n_values 11  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 10_000_000 --SEED $SEED
    python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project additive-penalty --group temp --algorithm ppo --start_range 0 --end_range 2 --n_values 11  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 10_000_000 --SEED $SEED
    python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project additive-penalty --group rh --algorithm ppo --start_range 0 --end_range 1 --n_values 11  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 10_000_000 --SEED $SEED
done
