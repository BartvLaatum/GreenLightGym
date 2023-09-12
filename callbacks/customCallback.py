import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from typing import Optional, Union
from stable_baselines3.common.vec_env import VecEnv, is_vecenv_wrapped


class TensorboardCallback(EvalCallback):

    """
    Callback that logs specified data from RL model and environment to Tensorboard.
    Is a daughter class of EvalCallback from Stable Baselines3.
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            super()._on_step()
