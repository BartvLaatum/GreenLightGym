# GreenLight-Gym: A Reinforcement Learning Environment for Greenhouse Crop Production Systems 🍅

![GreenLight-Gym Architecture|200](sketches/GLGymArchitecture.png)

## Summary

**This repository is a reimplementation of the high-tech greenhouse model [GreenLight](https://github.com/davkat1/GreenLight) in high-performance [Cython](https://cython.readthedocs.io/en/stable/index.html) code, wrapped by the Gymnasium environment. The environment is desinged to train reinforcement learning models for greenhouse crop production systems.**

The software has a modular architecture allowing users to study different aspects of the greenhouse control problem. For instance, one can customize:

- **Controllers** 
- **Controllable inputs**
- **Weather trajectories**
- **GreenLight model parameters**
- **Observations spaces**
- **Constraints & rewards**

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

3. Subsequently, install this project as an development package as follows:

```shell
pip install -e .
```

This allows you to make modifications to the greenlight_gym codebase without having to reinstall the complete package.

4. Since this the GreenLight model is built using Cython, one must recompile the Cython code. This can be achieved by:

```shell
pyhton setup.py build_ext --inplace
```

Which lets you rebuilt the GreenLight model without reinstalling the complete package again. Everytime you make adjustments to the Cython code one must recompile the code using the command.

## Usage

## Run commands for training RL algorithms


## Future updates

- Add environment based on setpoints instead of climate actuators

