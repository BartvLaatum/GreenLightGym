import os
import yaml
from os.path import join
from typing import Dict, Any, Callable, List

import wandb
from wandb.integration.sb3 import WandbCallback

from torch.optim import Adam
from torch.nn.modules.activation import ReLU, SiLU
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, VecEnv

from RLGreenLight.environments.GreenLight import GreenLight
from RLGreenLight.callbacks.customCallback import TensorboardCallback, SaveVecNormalizeCallback, BaseCallback

ACTIVATION_FN = {"ReLU": ReLU, "SiLU": SiLU}
OPTIMIZER = {"ADAM": Adam}

def loadParameters(env_id: str, path: str, filename: str, algorithm: str = None):
    with open(join(path, filename), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    envParams = params[env_id]
    options = params["options"]
    
    if algorithm is not None:
        modelParams = params[algorithm]

        if "policy_kwargs" in modelParams.keys():
            modelParams["policy_kwargs"]["activation_fn"] = \
                ACTIVATION_FN[modelParams["policy_kwargs"]["activation_fn"]]
            modelParams["policy_kwargs"]["optimizer_class"] = \
                OPTIMIZER[modelParams["policy_kwargs"]["optimizer_class"]]
    else:
        modelParams = None
    return envParams, modelParams, options

def wandb_init(modelParams: Dict[str, Any],
               envParams: Dict[str, Any],
               options: Dict[str, Any],
               timesteps: int,
               SEED: int,
               project: str,
               group: str,
               job_type: str,
               save_code: bool = False,
               resume: bool = False
               ):
    config= {
        "policy": modelParams["policy"],
        "total_timesteps": timesteps,
        "env": lambda: GreenLight(**envParams, options=options),
        "eval_env": lambda: GreenLight(**envParams, options=options, training=False),
        "seed": SEED,
        "note": "testing co2 control, daily balance",
        "modelParams": {**modelParams},
        "envParams": {**envParams}
    }

    run = wandb.init(
        project=project,
        config=config,
        group=group,
        sync_tensorboard=True,
        job_type=job_type,
        save_code=save_code,
        resume=True,
    )
    return run, config

def make_vec_env(env_fn: Callable, numCpus: int, monitor_filename: str = None, vec_norm_kwargs: Dict[str, Any] = None, eval_env: bool = False) -> VecEnv:
    """
    Creates a normalized environment.
    """
    # make dir if not exists
    if monitor_filename is not None and not os.path.exists(os.path.dirname(monitor_filename)):
        os.makedirs(os.path.dirname(monitor_filename), exist_ok=True)

    env = SubprocVecEnv([env_fn for _ in range(numCpus)])
    env = VecMonitor(env, filename=monitor_filename)
    env = VecNormalize(env, **vec_norm_kwargs)
    if eval_env:
        env.training = False
        env.norm_reward = False
    return env

def create_callbacks(eval_freq: int,
                     env_log_dir: str,
                     save_name: str,
                     model_log_dir: str,
                     eval_env: VecEnv,
                     verbose: int = 1,
                     ) -> List[BaseCallback]:
    save_vec_norm = SaveVecNormalizeCallback(save_freq=eval_freq, save_path=env_log_dir, name_prefix=save_name)
    save_vec_best = SaveVecNormalizeCallback(save_freq=1, save_path=env_log_dir, verbose=2)
    eval_callback = TensorboardCallback(eval_env, eval_freq=eval_freq, best_model_save_path=model_log_dir, name_vec_env=save_name, path_vec_env=env_log_dir, deterministic=True, callback_on_new_best=save_vec_best, verbose=verbose)
    wandbcallback = WandbCallback(verbose=verbose)
    return [save_vec_norm, eval_callback, wandbcallback]
