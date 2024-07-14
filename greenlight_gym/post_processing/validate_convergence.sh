#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Run the Python script

python post_processing/validate_model.py --project benchmark --group multiplicative-0.99 --runname elegant-carrier-2 --validation_type train --best_or_last last --n_eval_episodes 120 --save_results --env_id GreenLightHeatCO2
python post_processing/validate_model.py --project benchmark --group multiplicative-0.99 --runname rogue-podracer-3 --validation_type train --best_or_last last --n_eval_episodes 120 --save_results --env_id GreenLightHeatCO2
python post_processing/validate_model.py --project benchmark --group multiplicative-0.99 --runname ancient-tie-fighter-4 --validation_type train --best_or_last last --n_eval_episodes 120 --save_results --env_id GreenLightHeatCO2
python post_processing/validate_model.py --project benchmark --group multiplicative-0.99 --runname clean-pyramid-7 --validation_type train --best_or_last last --n_eval_episodes 120 --save_results --env_id GreenLightHeatCO2
python post_processing/validate_model.py --project benchmark --group multiplicative-0.99 --runname sweet-energy-9 --validation_type train --best_or_last last --n_eval_episodes 120 --save_results --env_id GreenLightHeatCO2
