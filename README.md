# GreenLight-Gym: A Reinforcement Learning Environment for Greenhouse Crop Production Systems 🍅

> Insert Image of the architecture?

<!-- ![hello](figures/GLGymArchitecture.pdf) -->

<object type="application/pdf" width="700px" height="700px">
    <embed src="figures/GLGymArchitecture.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="figures/GLGymArchitecture.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Summary

This repository is a reimplementation of the high-tech greenhouse model [GreenLight](https://github.com/davkat1/GreenLight) in high-performance Cython code, wrapped by the Gymnasium environment. The environment is desinged to train reinforcement learning models for greenhouse crop production systems. 

The software has a modular architecture allowing users to study different aspects of the greenhouse control problem. For instance, one can customize:
- Controller types 
- Controllable inputs 
- Weather disturbances 
- GreenLight model parameters
- Observation spaces
- Reward functions

## Installation
We recommend Python 3.8+ installation using [Anaconda](https://www.anaconda.com/).

Create a virtual conda environment as follows: 

```shell
conda create -n greenlight_gym python==3.11
```

and activate the environment:

```shell
conda activate greenlight_gym
```

Next, clone this repository

```shell
git clone git@github.com:BartvLaatum/GreenLightGym.git
```

and navigate into this folder

```shell
cd GreenLightGym
```

Where you will have to install the greenhouse environment.

## Usage


## Run commands for reproducing article figures

