# GreenLight-Gym: A Reinforcement Learning Environment for Greenhouse Crop Production Systems 🍅

![GreenLight-Gym Architecture](sketches/GLGymArchitecture.png)

## Summary

This repository is a reimplementation of the high-tech greenhouse model [GreenLight](https://github.com/davkat1/GreenLight) in high-performance [Cython](https://cython.readthedocs.io/en/stable/index.html) code, wrapped by the Gymnasium environment. The environment is desinged to train reinforcement learning models for greenhouse crop production systems. 

The software has a modular architecture allowing users to study different aspects of the greenhouse control problem. For instance, one can customize:
- Controller types 
- Controllable inputs 
- Weather disturbances 
- GreenLight model parameters
- Observations
- Constraints & rewards

## Installation
To be able to compile Cython code, and convert it into a C script, Windows user are required to Miscrosoft Visual Studio with Desktop C++ development. Also see this blog post over [here](https://stackoverflow.com/questions/60322655/how-to-use-cython-on-windows-10-with-python-3-8).

We recommend Python 3.8+ installation using [Anaconda](https://www.anaconda.com/).

1. Create a virtual conda environment as follows: 

```shell
conda create -n greenlight_gym python==3.11
```

and activate the environment:

```shell
conda activate greenlight_gym
```

2. Next, clone this repository

```shell
git clone git@github.com:BartvLaatum/GreenLightGym.git
```

and navigate into this folder

```shell
cd GreenLightGym
```

3. Subsequently, install the required dependencies using:

```shell
pip install -r requirements.txt
```

## Usage


## Run commands for reproducing article figures

