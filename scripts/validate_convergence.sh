#!/bin/bash

# Change to the directory containing the Python script
cd ./GreenLightGym/

# Run the Python script
python -m greenlight_gym.post_processing.validate_model --project benchmark --group multiplicative-0.99 --runname elegant-carrier-2 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2
python -m greenlight_gym.post_processing.validate_model --project benchmark --group multiplicative-0.99 --runname rogue-podracer-3 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2
python -m greenlight_gym.post_processing.validate_model --project benchmark --group multiplicative-0.99 --runname ancient-tie-fighter-4 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2
python -m greenlight_gym.post_processing.validate_model --project benchmark --group multiplicative-0.99 --runname clean-pyramid-7 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2
python -m greenlight_gym.post_processing.validate_model --project benchmark --group multiplicative-0.99 --runname sweet-energy-9 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2

python -m greenlight_gym.post_processing.validate_model --project benchmark --group additive-0.99 --runname vibrant-rain-15 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2
python -m greenlight_gym.post_processing.validate_model --project benchmark --group additive-0.99 --runname trim-dawn-16 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2
python -m greenlight_gym.post_processing.validate_model --project benchmark --group additive-0.99 --runname vague-breeze-17 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2
python -m greenlight_gym.post_processing.validate_model --project benchmark --group additive-0.99 --runname cerulean-dawn-19 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2
python -m greenlight_gym.post_processing.validate_model --project benchmark --group additive-0.99 --runname helpful-sky-20 --validation_type test --best_or_last last --n_eval_episodes 60 --save_results --env_id GreenLightHeatCO2
