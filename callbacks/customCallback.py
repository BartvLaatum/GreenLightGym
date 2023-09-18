import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union
from stable_baselines3.common.vec_env import VecEnv, is_vecenv_wrapped


class TensorboardCallback(EvalCallback):
    """
    Callback that logs specified data from RL model and environment to Tensorboard.
    Is a daughter class of EvalCallback from Stable Baselines3.
    Saves the best model according to the mean reward on a evaluation environment.
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        path_vec_env: Optional[str] = None,         # from where to load in VecNormalize
        name_vec_env: Optional[str] = None,         # name of the VecNormalize file
        callback_on_new_best = None,                # callback to call when a new best model is found
        verbose: int = 1,
        save_vec_normalize = None,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            callback_on_new_best=callback_on_new_best,
            verbose=verbose,
        )
        self.path_vec_env = path_vec_env
        self.name_vec_env = name_vec_env

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            path = os.path.join(self.path_vec_env, f"{self.name_vec_env}_{self.num_timesteps}_steps.pkl")
            self.eval_env = VecNormalize.load(path, self.eval_env)

            super()._on_step()

class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps
    FROM STABLE BASELINES3 RLZOO LIBRARY: 
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/f6b3ff70b13d2c2156b3e0faf9994c107c649c82/utils/callbacks.py#L55
    
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("-----------------------")
                    print(f"Saving VecNormalize to {path}")
                    print("-----------------------")

        return True
