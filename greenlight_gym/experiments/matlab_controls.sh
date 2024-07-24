#!/bin/bash

# Change to the directory containing the Python script
cd /home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Export the path as an environment variable so it's available in parallel executions
export PYTHONPATH=/home/bart/Documents/phd-code-projects/GreenLightGym/greenlight_gym

# Use GNU parallel to run the Python script in parallel
# parallel python experiments/matlab_controls.py --step_size {1} --date {2} --save ::: '0.5s' '1.0s' '2.0s' ::: '20000101' '20000201' '20000301' '20000401' '20000501' '20000601' '20000701' '20000801' '20000901' '20001001' '20001101' '20001201'
# 

python experiments/matlab_controls.py --step_size "0.01s" --date "20000101" --n_days 1 --save #{1} --date {2} --save ::: '0.5s' '1.0s' '2.0s' ::: '20000101' '20000201' '20000301' '20000401' '20000501' '20000601' '20000701' '20000801' '20000901' '20001001' '20001101' '20001201'
