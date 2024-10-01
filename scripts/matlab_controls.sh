#!/bin/bash

# Change to the directory containing the Python script
cd ./GreenLightGym/


python -m greenlight_gym.experiments.matlab_controls --step_size "0.01s" --date "20000101" --n_days 1 --save #{1} --date {2} --save ::: '0.5s' '1.0s' '2.0s' ::: '20000101' '20000201' '20000301' '20000401' '20000501' '20000601' '20000701' '20000801' '20000901' '20001001' '20001101' '20001201'
