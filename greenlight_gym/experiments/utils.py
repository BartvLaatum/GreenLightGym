import os
import yaml
from os.path import join
from typing import Dict, Any, Callable, List, Optional, Union, Tuple
import warnings

import wandb
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.nn.modules.activation import ReLU, SiLU, Tanh, ELU
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, VecEnv

from greenlight_gym.envs.greenlight import GreenLightBase, GreenLightCO2, GreenLightHeatCO2
from greenlight_gym.common.callbacks import TensorboardCallback, SaveVecNormalizeCallback, BaseCallback

ACTIVATION_FN = {"ReLU": ReLU, "SiLU": SiLU, "Tanh":Tanh, "ELU": ELU}
OPTIMIZER = {"ADAM": Adam}

envs = {"GreenLightBase": GreenLightBase, "GreenLightCO2": GreenLightCO2, "GreenLightHeatCO2": GreenLightHeatCO2}

def make_env(env_id, rank, seed, kwargs, kwargsSpecific, options, eval_env):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :return: (Gym Environment) The gym environment
    """
    def _init():
        env = envs[env_id](**kwargsSpecific, **kwargs, options=options)
        if eval_env:
            env.training = False

        # call reset with seed due to new seeding syntax of gymnasium environments
        env.reset(seed+rank)
        env.action_space.seed(seed + rank)
        return env
    return _init

def loadParameters(env_id: str, path: str, filename: str, algorithm: str = None):
    with open(join(path, filename), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    if env_id != "GreenLightBase":
        envSpecificParams = params[env_id]
    else:
        envSpecificParams = {}

    envBaseParams = params["GreenLightBase"]
    options = params["options"]

    state_columns = params["state_columns"]
    action_columns = params["action_columns"]

    
    if algorithm is not None:
        modelParams = params[algorithm]

        if "policy_kwargs" in modelParams.keys():
            modelParams["policy_kwargs"]["activation_fn"] = \
                ACTIVATION_FN[modelParams["policy_kwargs"]["activation_fn"]]
            modelParams["policy_kwargs"]["optimizer_class"] = \
                OPTIMIZER[modelParams["policy_kwargs"]["optimizer_class"]]
            modelParams["policy_kwargs"]["log_std_init"] = \
                eval(modelParams["policy_kwargs"]["log_std_init"])
    else:
        modelParams = None
    return envBaseParams, envSpecificParams, modelParams, options, state_columns, action_columns

def set_model_params(config):
    modelParams = {}
    policy_kwargs = {}
    policy_kwargs['activation_fn'] = ACTIVATION_FN[config['activation_fn']]
    policy_kwargs['activation_fn'] = ACTIVATION_FN[config['activation_fn']]
    policy_kwargs['net_arch'] = {"pi": [config["pi_size"]]*2, "vf": [config["vf_size"]]*2}
    policy_kwargs['optimizer_class'] = OPTIMIZER[config["optimizer_class"]]
    policy_kwargs['optimizer_kwargs'] = config['optimizer_kwargs']
    policy_kwargs['log_std_init'] = np.log(config['std_init'])

    modelParams["policy_kwargs"] = policy_kwargs

    modelParams['batch_size'] = config['batch_size']
    modelParams['n_steps'] = config['n_steps']
    modelParams['n_epochs'] = config['n_epochs']
    modelParams['learning_rate'] = config['learning_rate']
    modelParams['gamma'] = config['gamma']
    modelParams['gae_lambda'] = config['gae_lambda']
    modelParams['policy'] = config['policy']
    modelParams['normalize_advantage'] = config['normalize_advantage']
    modelParams['ent_coef'] = config['ent_coef']
    modelParams['vf_coef'] = config['vf_coef']
    modelParams['max_grad_norm'] = config['max_grad_norm']
    modelParams['use_sde'] = config['use_sde']
    modelParams['sde_sample_freq'] = config['sde_sample_freq']
    modelParams['target_kl'] = None

    return modelParams


def wandb_init(modelParams: Dict[str, Any],
               envParams: Dict[str, Any],
               envSpecificParams: Dict[str, Any],
               timesteps: int,
               SEED: int,
               project: str,
               group: str,
               runname: str,
               job_type: str,
               save_code: bool = False,
               resume: bool = False
               ):

    config= {
        "policy": modelParams["policy"],
        "total_timesteps": timesteps,
        "seed": SEED,
        "modelParams": {**modelParams},
        "envParams": {**envSpecificParams, **envParams}
    }
    config_exclude_keys = []
    run = wandb.init(
        project=project,
        config=config,
        group=group,
        name=runname,
        sync_tensorboard=True,
        config_exclude_keys=config_exclude_keys,
        job_type=job_type,
        save_code=save_code,
        resume=resume,
    )
    return run, config

def make_vec_env(env_id: str,
                 envParams: Dict[str, Any],
                 envSpecificParams: Dict[str, Any],
                 options: Dict[str, Any],
                 seed: int,
                 numCpus: int,
                 monitor_filename: str | None = None,
                 vec_norm_kwargs: Dict[str, Any] | None = None,
                 eval_env: bool = False) -> VecEnv:
    """
    Creates a normalized environment.
    """
    # make dir if not exists
    if monitor_filename is not None and not os.path.exists(os.path.dirname(monitor_filename)):
        os.makedirs(os.path.dirname(monitor_filename), exist_ok=True)

    env = SubprocVecEnv([make_env(env_id, rank, seed, envParams, envSpecificParams, options, eval_env=eval_env) for rank in range(numCpus)])
    env = VecMonitor(env, filename=monitor_filename)
    env = VecNormalize(env, **vec_norm_kwargs)
    env.seed(seed=seed)

    if eval_env:
        env.training = False
        env.norm_reward = False
    # env.seed(seed)
    return env

def create_callbacks(n_eval_episodes: int,
                     eval_freq: int,
                     env_log_dir: str,
                     save_name: str,
                     model_log_dir: str,
                     eval_env: VecEnv,
                     run: wandb.run = None,
                     action_columns: List[str] | None = None,
                     state_columns:List[str] | None = None,
                     states2plot:List[str] | None = None,
                     actions2plot: List[str] | None = None,
                     save_env: bool = True,
                     verbose: int = 1,
                     ) -> List[BaseCallback]:
    if save_env:
        save_vec_best = SaveVecNormalizeCallback(save_freq=1, save_path=env_log_dir, verbose=2)
    else:
        save_vec_best = None
    eval_callback = TensorboardCallback(eval_env,
                                        n_eval_episodes=n_eval_episodes,
                                        eval_freq=eval_freq,
                                        best_model_save_path=model_log_dir,
                                        name_vec_env=save_name,
                                        path_vec_env=env_log_dir,
                                        deterministic=True,
                                        callback_on_new_best=save_vec_best,
                                        run=run,
                                        action_columns=action_columns,
                                        state_columns=state_columns,
                                        states2plot=states2plot,
                                        actions2plot=actions2plot,
                                        verbose=verbose)
    wandbcallback = WandbCallback(verbose=verbose)
    return [eval_callback, wandbcallback]

def controlScheme(GL, nightValue, dayValue):
    """
    Function to test the effect of controlling a certain variable.
    """
    obs, info = GL.reset()
    GL.GLModel.setNightCo2(nightValue)
    N = GL.N                                        # number of timesteps to take
    states = np.zeros((N+1, GL.modelObsVars))       # array to save states
    controlSignals = np.zeros((N+1, GL.GLModel.nu)) # array to save rule-based controls controls
    states[0, :] = obs[:GL.modelObsVars]            # get initial states
    timevec = np.zeros((N+1,))                      # array to save time
    timevec[0] = GL.GLModel.time
    i=1

    while not GL.terminated:
        # check whether it is day or night
        if GL.weatherData[GL.GLModel.timestep * GL.solverSteps, 9] > 0:
            controls = np.ones((GL.action_space.shape[0],))*dayValue
        else:
            controls = np.ones((GL.action_space.shape[0],))*nightValue
        obs, r, terminated, _, info = GL.step(controls.astype(np.float32))
        states[i, :] += obs[:GL.modelObsVars]
        controlSignals[i, :] += info["controls"]
        timevec[i] = info["Time"]
        i+=1

    # insert time vector into states array
    states = np.insert(states, 0, timevec, axis=1)
    states = pd.DataFrame(data=states[:], columns=["Time", "Air Temperature", "CO2 concentration", "Humidity", "Fruit weight", "Fruit harvest", "PAR"])
    controlSignals = pd.DataFrame(data=controlSignals, columns=["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uIntLamp", "uGroPipe", "uBlScr"])
    weatherData = pd.DataFrame(data=GL.weatherData[[int(ts * GL.solverSteps) for ts in range(0, GL.Np+1)], :GL.weatherObsVars], columns=["Temperature", "Humidity", "PAR", "CO2 concentration", "Wind"])

    return states, controlSignals, weatherData

def runRuleBasedController(GL, stateColumns, actionColumns):
    obs, info = GL.reset()
    N = GL.N                                        # number of timesteps to take
    states = np.zeros((N+1, GL.modelObsVars))       # array to save states
    controlSignals = np.zeros((N+1, GL.GLModel.nu)) # array to save rule-based controls controls
    states[0, :] = obs[:GL.modelObsVars]             # get initial states
    timevec = np.zeros((N+1,))                      # array to save time
    timevec[0] = GL.GLModel.time
    i=1
    while not GL.terminated:
        controls = np.ones((GL.action_space.shape[0],))*0.5
        obs, r, terminated, _, info = GL.step(controls.astype(np.float32))
        states[i, :] += obs[:GL.modelObsVars]
        controlSignals[i, :] += info["controls"]
        timevec[i] = info["Time"]
        i+=1
    
    # insert time vector into states array
    states = np.insert(states, 0, timevec, axis=1)
    states = pd.DataFrame(data=states[:], columns=["Time"]+stateColumns)
    controlSignals = pd.DataFrame(data=controlSignals, columns=actionColumns)
    weatherData = pd.DataFrame(data=GL.weatherData[[int(ts * GL.timeinterval/GL.h) for ts in range(0, GL.Np+1)], :GL.weatherObsVars], columns=["Temperature", "Humidity", "PAR", "CO2 concentration", "Wind"])

    return states, controlSignals, weatherData

def runSimulationDefinedControls(GL, matlabControls, stateNames, matlabStates, nx):
    obs, info = GL.reset()
    N = matlabControls.shape[0]

    cythonStates = np.zeros((N, nx))
    cyhtonControls = np.zeros((N, GL.GLModel.nu))
    cythonStates[0, :] = GL.GLModel.getStatesArray()

    for i in range(1, N):
        # print(i)
        controls = matlabControls.iloc[i, :].values
        obs, reward, terminated, truncated, info = GL.step(controls)
        cythonStates[i, :] += GL.GLModel.getStatesArray()
        cyhtonControls[i, :] += info["controls"]
        # print("Day of the year", GL.GLModel.dayOfYear)
        # print("time since midnight in hours", GL.GLModel.timeOfDay)
        # print("Time lamp of the day", GL.GLModel.lampTimeOfDay)

        if terminated:
            break

    cythonStates = pd.DataFrame(data=cythonStates, columns=stateNames[:])
    cyhtonControls = pd.DataFrame(data=cyhtonControls, columns=["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uIntLamp", "uGroPipe", "uBlScr", "shScr", "perShScr", "uSide"])
    return cythonStates, cyhtonControls
