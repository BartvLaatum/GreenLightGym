#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Run the Python script
python experiments/train_agent.py --env_id GreenLightHeatCO2
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project 1st-half-year --group ent-coef-0.01 --HPfolder gl_heat_co2 --total_timesteps 2_000_000 --seed 667
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project 1st-half-year --group ent-coef-0.01 --HPfolder gl_heat_co2 --total_timesteps 2_000_000 --seed 668
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project 1st-half-year --group ent-coef-0.01 --HPfolder gl_heat_co2 --total_timesteps 2_000_000 --seed 669
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project 1st-half-year --group ent-coef-0.01 --HPfolder gl_heat_co2 --total_timesteps 2_000_000 --seed 670
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project optimized-hyperparams --group no-low-co2-pen --HPfolder gl_heat_co2 --total_timesteps 2_000_000 --seed 668
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project optimized-hyperparams --group no-low-co2-pen --HPfolder gl_heat_co2 --total_timesteps 2_000_000 --seed 669
# python experiments/train_agent.py --env_id GreenLightHeatCO2 --project optimized-hyperparams --group no-low-co2-pen --HPfolder gl_heat_co2 --total_timesteps 2_000_000 --seed 670
