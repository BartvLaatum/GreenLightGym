import os
import yaml
from os.path import join
from typing import Dict, Any, List, Optional, Tuple

import wandb
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.nn.modules.activation import ReLU, SiLU, Tanh, ELU
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, VecEnv

from greenlight_gym.common.results import Results
from greenlight_gym.common.callbacks import TensorboardCallback, SaveVecNormalizeCallback, BaseCallback
from greenlight_gym.envs.greenlight import GreenLightEnv, GreenLightHeatCO2, GreenLightRuleBased, GreenLightStatesTest
from greenlight_gym.common.learning_rate import linear_schedule

ACTIVATION_FN = {"ReLU": ReLU, "SiLU": SiLU, "Tanh":Tanh, "ELU": ELU}
OPTIMIZER = {"ADAM": Adam}

envs = {"GreenLightEnv": GreenLightEnv, "GreenLightHeatCO2": GreenLightHeatCO2, "GreenLightRuleBased": GreenLightRuleBased,
        "GreenLightStatesTest": GreenLightStatesTest}

def make_env(env_id, rank, seed, kwargs, kwargsSpecific, options, eval_env):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :return: (Gym Environment) The gym environment
    """
    def _init():
        env = envs[env_id](**kwargsSpecific, **kwargs)
        if eval_env:
            env.training = False
            env.start_days = options["start_days"]
            # we fix the growth year for each individual evaluation environment
            env.growth_year = options["growth_years"][rank]

        # call reset with seed due to new seeding syntax of gymnasium environments
        env.reset(seed+rank)
        env.action_space.seed(seed + rank)
        return env
    return _init

def load_model_params(algorithm: str, path: str, env_name: str) -> Dict[str, Any]:

    with open(join(path, algorithm + ".yml"), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    model_params = params[env_name]

    if "policy_kwargs" in model_params.keys():
        model_params["policy_kwargs"]["activation_fn"] = \
            ACTIVATION_FN[model_params["policy_kwargs"]["activation_fn"]]
        model_params["policy_kwargs"]["optimizer_class"] = \
            OPTIMIZER[model_params["policy_kwargs"]["optimizer_class"]]
        model_params["policy_kwargs"]["log_std_init"] = \
            eval(model_params["policy_kwargs"]["log_std_init"])
        
    if "learning_rate_scheduler" in model_params.keys():
        model_params["learning_rate"] = linear_schedule(**model_params["learning_rate_scheduler"])
        del model_params["learning_rate_scheduler"]
    return model_params

def load_env_params(env_id: str, path: str, filename: str) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """"
    Function that loads in the environment variables. 
    Returns the variables for the general parent GreenLightEnv class,
    if one aims to use specified environment these are also returned.
    Also, options, action and state columns are returned.
    """
    with open(join(path, filename + ".yml"), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if env_id != "GreenLightEnv":
        env_specific_params = params[env_id]
    else:
        env_specific_params = {}

    env_base_params = params["GreenLightEnv"]

    # options dictionary specifying the evaluation environment 
    options = params["options"]

    # state and action column names used for plotting and saving data
    results_columns = params["results_columns"]
    return env_base_params, env_specific_params, options, results_columns

def loadParameters(env_id: str, path: str, filename: str, algorithm: str = None):
    with open(join(path, filename), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    if env_id != "GreenLightEnv":
        envSpecificParams = params[env_id]
    else:
        envSpecificParams = {}

    envBaseParams = params["GreenLightEnv"]
    options = params["options"]

    state_columns = params["state_columns"]
    action_columns = params["action_columns"]

    
    if algorithm is not None:
        model_params = params[algorithm]

        if "policy_kwargs" in model_params.keys():
            model_params["policy_kwargs"]["activation_fn"] = \
                ACTIVATION_FN[model_params["policy_kwargs"]["activation_fn"]]
            model_params["policy_kwargs"]["optimizer_class"] = \
                OPTIMIZER[model_params["policy_kwargs"]["optimizer_class"]]
            model_params["policy_kwargs"]["log_std_init"] = \
                eval(model_params["policy_kwargs"]["log_std_init"])
    else:
        model_params = None
    return envBaseParams, envSpecificParams, model_params, options, state_columns, action_columns

def set_model_params(config):
    model_params = {}
    policy_kwargs = {}
    policy_kwargs['activation_fn'] = ACTIVATION_FN[config['activation_fn']]
    policy_kwargs['activation_fn'] = ACTIVATION_FN[config['activation_fn']]
    policy_kwargs['net_arch'] = {"pi": [config["pi_size"]]*2, "vf": [config["vf_size"]]*2}
    policy_kwargs['optimizer_class'] = OPTIMIZER[config["optimizer_class"]]
    policy_kwargs['optimizer_kwargs'] = config['optimizer_kwargs']
    policy_kwargs['log_std_init'] = np.log(config['std_init'])

    model_params["policy_kwargs"] = policy_kwargs

    model_params['batch_size'] = config['batch_size']
    model_params['n_steps'] = config['n_steps']
    model_params['n_epochs'] = config['n_epochs']
    model_params['learning_rate'] = config['learning_rate']
    model_params['gamma'] = config['gamma']
    model_params['gae_lambda'] = config['gae_lambda']
    model_params['policy'] = config['policy']
    model_params['normalize_advantage'] = config['normalize_advantage']
    model_params['ent_coef'] = config['ent_coef']
    model_params['vf_coef'] = config['vf_coef']
    model_params['max_grad_norm'] = config['max_grad_norm']
    model_params['use_sde'] = config['use_sde']
    model_params['sde_sample_freq'] = config['sde_sample_freq']
    model_params['target_kl'] = None

    return model_params


def wandb_init(model_params: Dict[str, Any],
               env_params: Dict[str, Any],
               env_specific_params: Dict[str, Any],
               timesteps: int,
               env_seed: int,
               model_seed: int,
               project: str,
               group: str,
               runname: str,
               job_type: str,
               save_code: bool = False,
               resume: bool = False
               ):

    config= {
        "policy": model_params["policy"],
        "total_timesteps": timesteps,
        "env_seed": env_seed,
        'model_seed': model_seed,
        "model_params": {**model_params},
        "env_params": {**env_specific_params, **env_params}
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
                 n_envs: int,
                 monitor_filename: str | None = None,
                 vec_norm_kwargs: Dict[str, Any] | None = None,
                 eval_env: bool = False) -> VecEnv:
    """
    Creates a vectorized environment, with n individual envs.
    """
    # make dir if not exists
    if monitor_filename is not None and not os.path.exists(os.path.dirname(monitor_filename)):
        os.makedirs(os.path.dirname(monitor_filename), exist_ok=True)

    env = SubprocVecEnv([make_env(env_id, rank, seed, envParams, envSpecificParams, options, eval_env=eval_env) for rank in range(n_envs)])
    env = VecMonitor(env, filename=monitor_filename)
    env = VecNormalize(env, **vec_norm_kwargs)
    env.seed(seed=seed)
    if eval_env:
        env.training = False
        env.norm_reward = False
    return env

def create_callbacks(n_eval_episodes: int,
                     eval_freq: int,
                     env_log_dir: str,
                     save_name: str,
                     model_log_dir: str,
                     eval_env: VecEnv,
                     run: wandb.run = None,
                     results: Optional[Results] = None,
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
                                        results=results,
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
