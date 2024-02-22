#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project penalty-coeffs-3 --group co2 --algorithm ppo --start_range 0 --end_range 1e-2 --n_values 11  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 2_000_000 --SEED 666
# python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project penalty-coeffs-3 --group temp --algorithm ppo --start_range 0 --end_range 2 --n_values 11  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 2_000_000 --SEED 666
# python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project penalty-coeffs-3 --group rh --algorithm ppo --start_range 0 --end_range 1 --n_values 11  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 2_000_000 --SEED 666

python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project penalty-coeffs-3 --group co2 --algorithm ppo --start_range 0 --end_range 1e-2 --n_values 11  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 2_000_000 --SEED 667
python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project penalty-coeffs-3 --group temp --algorithm ppo --start_range 0 --end_range 2 --n_values 11  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 2_000_000 --SEED 667
python experiments/penalty_coeffs.py --env_id GreenLightHeatCO2 --project penalty-coeffs-3 --group rh --algorithm ppo --start_range 0 --end_range 1 --n_values 11  --n_evals 1 --num_cpus 12 --n_eval_episodes 60 --total_timestep 2_000_000 --SEED 667
